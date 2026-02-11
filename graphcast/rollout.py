# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for rolling out models."""

from typing import Iterator, Optional, Sequence

from absl import logging
import chex
import dask.array
from graphcast import xarray_jax
from graphcast import xarray_tree
import jax
import numpy as np
import typing_extensions
import xarray


class PredictorFn(typing_extensions.Protocol):
  """Functional version of base.Predictor.__call__ with explicit rng."""

  def __call__(
      self, rng: chex.PRNGKey, inputs: xarray.Dataset,
      targets_template: xarray.Dataset,
      forcings: xarray.Dataset,
      **optional_kwargs,
      ) -> xarray.Dataset:
    ...


def _replicate_dataset(
    data: xarray.Dataset, replica_dim: str,
    replicate_to_device: bool,
    devices: Sequence[jax.Device],
    ) -> xarray.Dataset:
  """Used to prepare for xarray_jax.pmap."""

  def replicate_variable(variable: xarray.Variable) -> xarray.Variable:
    if replica_dim in variable.dims:
      # TODO(pricei): call device_put_replicated when replicate_to_device==True
      return variable.transpose(replica_dim, ...)
    else:
      data = len(devices) * [variable.data]
      if replicate_to_device:
        assert devices is not None
        # TODO(pricei): Refactor code to use "device_put_replicated" instead of
        # device_put_sharded.
        data = jax.device_put_sharded(data, devices)
      else:
        data = np.stack(data, axis=0)
      return xarray_jax.Variable(
          data=data, dims=(replica_dim,) + variable.dims, attrs=variable.attrs
      )

  def replicate_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    if dataset is None:
      return None
    data_variables = {
        name: replicate_variable(var)
        for name, var in dataset.data_vars.variables.items()
    }
    coords = {name: coord.variable for name, coord in dataset.coords.items()}
    return xarray.Dataset(data_variables, coords=coords, attrs=dataset.attrs)

  return replicate_dataset(data)


def chunked_prediction_generator_multiple_runs(
    predictor_fn: PredictorFn,
    rngs: chex.PRNGKey,
    inputs: xarray.Dataset,
    targets_template: xarray.Dataset,
    forcings: Optional[xarray.Dataset],
    num_samples: Optional[int],
    pmap_devices: Optional[Sequence[jax.Device]] = None,
    **chunked_prediction_kwargs,
) -> Iterator[xarray.Dataset]:
  """Outputs a trajectory of multiple samples by yielding chunked predictions.

  Args:
    predictor_fn: Function to use to make predictions for each chunk.
    rngs: RNG sequence to be used for each ensemble member.
    inputs: Inputs for the model.
    targets_template: Template for the target prediction, requires targets
        equispaced in time.
    forcings: Optional forcing for the model.
    num_samples: The number of runs / samples to rollout.
    pmap_devices: List of devices over which predictor_fn is pmapped, or None if
      it is not pmapped.
    **chunked_prediction_kwargs:
      See chunked_prediction, some of these are required arguments.

  Yields:
    The predictions for each chunked step of the chunked rollout, such that
    if all predictions are concatenated in time and sample dimension squeezed,
    this would match the targets template in structure.

  """
  if pmap_devices is not None:
    assert (
        num_samples % len(pmap_devices) == 0
    ), "num_samples must be a multiple of len(pmap_devices)"

    def predictor_fn_pmap_named_args(rng, inputs, targets_template, forcings):
      targets_template = _replicate_dataset(
          targets_template,
          replica_dim="sample",
          replicate_to_device=True,
          devices=pmap_devices,
      )
      return predictor_fn(rng, inputs, targets_template, forcings)

    for i in range(0, num_samples, len(pmap_devices)):
      sample_idx = slice(i, i + len(pmap_devices))
      logging.info("Samples %s out of %s", sample_idx, num_samples)
      logging.flush()
      sample_group_rngs = rngs[sample_idx]

      if "sample" not in inputs.dims:
        sample_inputs = inputs
      else:
        sample_inputs = inputs.isel(sample=sample_idx, drop=True)

      sample_inputs = _replicate_dataset(
          sample_inputs,
          replica_dim="sample",
          replicate_to_device=True,
          devices=pmap_devices,
      )

      if forcings is not None:
        if "sample" not in forcings.dims:
          sample_forcings = forcings
        else:
          sample_forcings = forcings.isel(sample=sample_idx, drop=True)

        # TODO(pricei): We are replicating the full forcings for all rollout
        # timesteps here, rather than inside `predictor_fn_pmap_named_args` like
        # the targets_template above, because the forcings are concatenated with
        # the inputs which will already be replicated. We should refactor this
        # so that chunked prediction is aware of whether it is being run with
        # pmap, and if so do the replication and device_put only of the
        # necessary timesteps, as part of the chunked prediction function.
        sample_forcings = _replicate_dataset(
            sample_forcings,
            replica_dim="sample",
            replicate_to_device=False,
            devices=pmap_devices,
        )
      else:
        sample_forcings = None

      for prediction_chunk in chunked_prediction_generator(
          predictor_fn=predictor_fn_pmap_named_args,
          rng=sample_group_rngs,
          inputs=sample_inputs,
          targets_template=targets_template,
          forcings=sample_forcings,
          pmap_devices=pmap_devices,
          **chunked_prediction_kwargs,
      ):
        prediction_chunk.coords["sample"] = np.arange(
            sample_idx.start, sample_idx.stop, sample_idx.step
        )
        yield prediction_chunk
        del prediction_chunk
  else:
    for i in range(num_samples):
      logging.info("Sample %d/%d", i, num_samples)
      logging.flush()
      this_sample_rng = rngs[i]

      if "sample" in inputs.dims:
        sample_inputs = inputs.isel(sample=i, drop=True)
      else:
        sample_inputs = inputs

      sample_forcings = forcings
      if sample_forcings is not None:
        if "sample" in sample_forcings.dims:
          sample_forcings = sample_forcings.isel(sample=i, drop=True)

      for prediction_chunk in chunked_prediction_generator(
          predictor_fn=predictor_fn,
          rng=this_sample_rng,
          inputs=sample_inputs,
          targets_template=targets_template,
          forcings=sample_forcings,
          **chunked_prediction_kwargs):
        prediction_chunk.coords["sample"] = i
        yield prediction_chunk
        del prediction_chunk


def chunked_prediction(
    predictor_fn: PredictorFn,
    rng: chex.PRNGKey,
    inputs: xarray.Dataset,
    targets_template: xarray.Dataset,
    forcings: xarray.Dataset,
    num_steps_per_chunk: int = 1,
    verbose: bool = False,
    **chunked_prediction_kwargs,
) -> xarray.Dataset:
  """Outputs a long trajectory by iteratively concatenating chunked predictions.

  Args:
    predictor_fn: Function to use to make predictions for each chunk.
    rng: Random key.
    inputs: Inputs for the model.
    targets_template: Template for the target prediction, requires targets
        equispaced in time.
    forcings: Optional forcing for the model.
    num_steps_per_chunk: How many of the steps in `targets_template` to predict
        at each call of `predictor_fn`. It must evenly divide the number of
        steps in `targets_template`.
    verbose: Whether to log the current chunk being predicted.
    **chunked_prediction_kwargs:
      Extra arguments forwarded to `chunked_prediction_generator`
      (e.g., truth_ds, truth_t_path, temp_var_name, inject_from_step).

  Returns:
    Predictions for the targets template.

  """
  chunks_list = []
  for prediction_chunk in chunked_prediction_generator(
      predictor_fn=predictor_fn,
      rng=rng,
      inputs=inputs,
      targets_template=targets_template,
      forcings=forcings,
      num_steps_per_chunk=num_steps_per_chunk,
      verbose=verbose,
      **chunked_prediction_kwargs):
    chunks_list.append(jax.device_get(prediction_chunk))
  return xarray.concat(chunks_list, dim="time")


from typing import Iterator, Optional, Sequence
import logging
import numpy as np
import chex
import jax
import xarray as xr


def chunked_prediction_generator(
    predictor_fn,
    rng: chex.PRNGKey,
    inputs: xr.Dataset,
    targets_template: xr.Dataset,
    forcings: xr.Dataset,
    num_steps_per_chunk: int = 1,
    verbose: bool = False,
    pmap_devices: Optional[Sequence[jax.Device]] = None,

    # ---- U/V WIND INSERTION (LEAD-TIME MODE) ----
    truth_ds: Optional[xr.Dataset] = None,          # MUST have time=lead_times (timedeltas)
    u_var_name: str = "u_component_of_wind",        # variable name in truth_ds AND predictions (pressure-level U)
    v_var_name: str = "v_component_of_wind",        # variable name in truth_ds AND predictions (pressure-level V)
    inject_from_step: Optional[int] = None,         # 0-based global step index
) -> Iterator[xr.Dataset]:
  """
  Outputs a long trajectory by yielding chunked predictions.

  Lead-time mode contract:
  - targets_template.time is lead times (e.g. 6h, 12h, 18h ...)
  - truth_ds.time MUST ALSO be lead times (same dtype/values as targets_template.time)

  This version performs HARD truth insertion for:
    - u_var_name (U wind) at ALL pressure levels
    - v_var_name (V wind) at ALL pressure levels
  starting from inject_from_step, and the inserted values are used for subsequent AR steps.
  """

  # Make copies to avoid mutating callers.
  inputs = xr.Dataset(inputs)
  targets_template = xr.Dataset(targets_template)
  forcings = xr.Dataset(forcings)

  # Strip datetime coord (GraphCast does this to prevent recompiles)
  if "datetime" in inputs.coords:
    del inputs.coords["datetime"]

  if "datetime" in targets_template.coords:
    output_datetime = targets_template.coords["datetime"]
    del targets_template.coords["datetime"]
  else:
    output_datetime = None

  if "datetime" in forcings.coords:
    del forcings.coords["datetime"]

  # Basic shape checks
  if "time" not in targets_template.dims:
    raise ValueError("targets_template must have a 'time' dimension.")
  if "time" not in forcings.dims:
    raise ValueError("forcings must have a 'time' dimension.")
  if targets_template.sizes["time"] != forcings.sizes["time"]:
    raise ValueError(
        f"targets_template.time ({targets_template.sizes['time']}) and "
        f"forcings.time ({forcings.sizes['time']}) must match."
    )

  num_target_steps = targets_template.sizes["time"]
  num_chunks, remainder = divmod(num_target_steps, num_steps_per_chunk)
  if remainder != 0:
    raise ValueError(
        f"num_steps_per_chunk={num_steps_per_chunk} must evenly divide "
        f"num_target_steps={num_target_steps}"
    )

  # Targets time axis must be evenly spaced
  if len(np.unique(np.diff(targets_template.coords["time"].data))) > 1:
    raise ValueError("targets_template.time must be evenly spaced")

  # Template time used for ALL chunks (first chunk only) to avoid recompilation
  targets_chunk_time = targets_template.time.isel(time=slice(0, num_steps_per_chunk))

  current_inputs = inputs

  # ---- Validate truth dataset if insertion requested ----
  if inject_from_step is not None:
    if truth_ds is None:
      raise ValueError("inject_from_step is set but truth_ds is None (lead-time truth is required).")

    if "time" not in truth_ds.coords:
      raise ValueError("truth_ds must have a 'time' coordinate (lead times).")

    # Must contain both U and V
    missing = [vn for vn in (u_var_name, v_var_name) if vn not in truth_ds.data_vars]
    if missing:
      raise KeyError(
          f"Missing wind variables in truth_ds: {missing}. "
          f"Available: {list(truth_ds.data_vars)}"
      )

    if truth_ds.sizes.get("time", 0) < num_target_steps:
      raise ValueError(
          f"truth_ds has only {truth_ds.sizes.get('time', 0)} time steps but "
          f"targets_template needs {num_target_steps}."
      )

    if inject_from_step < 0 or inject_from_step >= num_target_steps:
      raise ValueError(
          f"inject_from_step={inject_from_step} out of range for num_target_steps={num_target_steps}."
      )

    # Strong alignment check: lead-times must match (for the horizon we use)
    tt = targets_template["time"].values
    tr = truth_ds["time"].values
    if not np.array_equal(tr[:num_target_steps], tt):
      raise ValueError(
          "truth_ds.time does not exactly match targets_template.time (lead-time mode).\n"
          f"targets_template.time[:5]={tt[:5]}\n"
          f"truth_ds.time[:5]={tr[:5]}\n"
          f"targets_template.time[-5:]={tt[-5:]}\n"
          f"truth_ds.time[-5:]={tr[:num_target_steps][-5:]}\n"
          "Fix by building truth_ds with time=lead_times exactly equal to targets_template.time."
      )

  # Global step counter (0..num_target_steps-1)
  global_step_offset = 0

  def split_rng_fn(rng_):
    rng1, rng2 = jax.random.split(rng_)
    return rng1, rng2

  if pmap_devices is not None:
    split_rng_fn = jax.pmap(split_rng_fn, devices=pmap_devices)

  for chunk_index in range(num_chunks):
    if verbose:
      logging.info("Chunk %d/%d", chunk_index + 1, num_chunks)

    target_offset = num_steps_per_chunk * chunk_index
    target_slice = slice(target_offset, target_offset + num_steps_per_chunk)

    current_targets_template = targets_template.isel(time=target_slice)

    # REAL lead-time labels for this chunk
    actual_target_time = current_targets_template.coords["time"]

    # Replace time coord with first-chunk time coord to avoid recompilation
    current_targets_template = current_targets_template.assign_coords(time=targets_chunk_time).compute()

    current_forcings = forcings.isel(time=target_slice)
    current_forcings = current_forcings.assign_coords(time=targets_chunk_time).compute()

    # Model call
    rng, this_rng = split_rng_fn(rng)
    predictions = predictor_fn(
        rng=this_rng,
        inputs=current_inputs,
        targets_template=current_targets_template,
        forcings=current_forcings
    )

    # Always bring to host before mutation
    predictions = jax.device_get(predictions)
    current_forcings = jax.device_get(current_forcings)
    current_inputs = jax.device_get(current_inputs)

    # ---- U/V WIND INSERTION (LEAD-TIME MODE) ----
    if inject_from_step is not None:
      # Ensure both exist in predictions
      missing_pred = [vn for vn in (u_var_name, v_var_name) if vn not in predictions.data_vars]
      if missing_pred:
        raise KeyError(
            f"Missing wind variables in predictions: {missing_pred}. "
            f"Available: {list(predictions.data_vars)}"
        )

      step_leads = actual_target_time.data
      global_steps = global_step_offset + np.arange(len(step_leads))

      mask = global_steps >= inject_from_step
      if np.any(mask):
        inject_idx = np.nonzero(mask)[0]
        leads_to_inject = step_leads[mask]

        # helper: inject one variable
        def _inject_one(var_name: str):
          pred_da = predictions[var_name]

          # Select truth for this chunk
          try:
            truth_chunk = truth_ds[var_name].sel(time=leads_to_inject)
          except Exception:
            truth_chunk = truth_ds[var_name].isel(time=global_steps[mask])

          # 1) Ensure truth has batch dim if predictions do
          if "batch" in pred_da.dims and "batch" not in truth_chunk.dims:
            truth_chunk = truth_chunk.expand_dims(batch=pred_da.coords["batch"])

          # 2) Align truth to prediction grid/dims for the injected timesteps
          truth_chunk = truth_chunk.reindex_like(
              pred_da.isel(time=inject_idx),
              method="nearest"
          )

          # 3) Assign per time index (integer indexing)
          for k, ti in enumerate(inject_idx):
            pred_da[dict(time=ti)] = truth_chunk.isel(time=k)

          predictions[var_name] = pred_da

        _inject_one(u_var_name)
        _inject_one(v_var_name)
    # -------------------- end U/V INSERTION --------------------

    # ---- Build next inputs (autoregressive state) ----
    if chunk_index < num_chunks - 1:
      next_frame = xr.merge([predictions, current_forcings])
      next_inputs = _get_next_inputs(current_inputs, next_frame)

      # shift time coords back for compilation stability
      next_inputs = next_inputs.assign_coords(time=current_inputs.coords["time"])
      current_inputs = next_inputs
    else:
      current_inputs = None

    global_step_offset += num_steps_per_chunk

    # Restore actual lead-time labels on predictions
    predictions = predictions.assign_coords(time=actual_target_time)

    if output_datetime is not None:
      predictions.coords["datetime"] = output_datetime.isel(time=target_slice)

    yield predictions
    del predictions



def _get_next_inputs(
    prev_inputs: xarray.Dataset, next_frame: xarray.Dataset,
    ) -> xarray.Dataset:
  """Computes next inputs, from previous inputs and predictions."""

  # Make sure are are predicting all inputs with a time axis.
  non_predicted_or_forced_inputs = list(
      set(prev_inputs.keys()) - set(next_frame.keys()))
  if "time" in prev_inputs[non_predicted_or_forced_inputs].dims:
    raise ValueError(
        "Found an input with a time index that is not predicted or forced.")

  # Keys we need to copy from predictions to inputs.
  next_inputs_keys = list(
      set(next_frame.keys()).intersection(set(prev_inputs.keys())))
  next_inputs = next_frame[next_inputs_keys]

  # Apply concatenate next frame with inputs, crop what we don't need.
  num_inputs = prev_inputs.dims["time"]
  return (
      xarray.concat(
          [prev_inputs, next_inputs], dim="time", data_vars="different")
      .tail(time=num_inputs))


def extend_targets_template(
    targets_template: xarray.Dataset,
    required_num_steps: int) -> xarray.Dataset:
  """Extends `targets_template` to `required_num_steps` with lazy arrays.

  It uses lazy dask arrays of zeros, so it does not require instantiating the
  array in memory.

  Args:
    targets_template: Input template to extend.
    required_num_steps: Number of steps required in the returned template.

  Returns:
    `xarray.Dataset` identical in variables and timestep to `targets_template`
    full of `dask.array.zeros` such that the time axis has `required_num_steps`.

  """

  # Extend the "time" and "datetime" coordinates
  time = targets_template.coords["time"]

  # Assert the first target time corresponds to the timestep.
  timestep = time[0].data
  if time.shape[0] > 1:
    assert np.all(timestep == time[1:] - time[:-1])

  extended_time = (np.arange(required_num_steps) + 1) * timestep

  if "datetime" in targets_template.coords:
    datetime = targets_template.coords["datetime"]
    extended_datetime = (datetime[0].data - timestep) + extended_time
  else:
    extended_datetime = None

  # Replace the values with empty dask arrays extending the time coordinates.
  datetime = targets_template.coords["time"]

  def extend_time(data_array: xarray.DataArray) -> xarray.DataArray:
    dims = data_array.dims
    shape = list(data_array.shape)
    shape[dims.index("time")] = required_num_steps
    dask_data = dask.array.zeros(
        shape=tuple(shape),
        chunks=-1,  # Will give chunk info directly to `ChunksToZarr``.
        dtype=data_array.dtype)

    coords = dict(data_array.coords)
    coords["time"] = extended_time

    if extended_datetime is not None:
      coords["datetime"] = ("time", extended_datetime)

    return xarray.DataArray(
        dims=dims,
        data=dask_data,
        coords=coords)

  return xarray_tree.map_structure(extend_time, targets_template)
