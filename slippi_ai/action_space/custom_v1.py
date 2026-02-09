"""Custom action space for Slippi AI.

L/R and X/Y are merged.

Shoulder is discretized into 3 levels: none, light shield, and full press.
Z pressed implies A and light shoulder.
Z, A, and shoulder are combined into a single feature with 7 buckets:

Notes:
The reason you'd want to (full) press analog but not digital is because the
analog -> digital transition causes your shield to drop for two frames. See
https://www.youtube.com/watch?v=gbBuvfWSzGw&t for an explanation.

The C-stick is bucketed in polar coordinates, with 13 buckets:
- 1 bucket for the origin (deadzone)
- 4 cardinal directions with small radius
- 8 intercardinal directions with large radius

The buttons and c-stick are combined into a single discrete action space with
32 * 13 = 416 total actions.

The main stick is bucketed separately in polar coordinates, with angle bucket
counts [1, 4, 16, 64] for each radius, totaling 85 buckets.
"""
import dataclasses
import enum
import typing as tp

import numpy as np

from slippi_ai.types import (
    Stick, Controller, Buttons,
    BoolArray, UInt8Array, UInt16Array, FloatArray, S,
)

class AnalogShoulder(enum.IntEnum):
  NONE = 0
  LIGHT = 1
  FULL = 2

# NOTE: this shoulder bucketing is pretty off for values around 0.7.
# We might want to add a new bucket between light and full.
LIGHT_SHOULDER_THRESHOLD = np.float32(0.3)
FULL_SHOULDER_THRESHOLD = np.float32(0.9)

def bucket_analog_shoulder(shoulder: FloatArray[S]) -> UInt8Array[S]:
  """Discretize shoulder input into buckets."""
  buckets = np.zeros_like(shoulder, dtype=np.uint8)
  buckets = tp.cast(UInt8Array[S], buckets)

  buckets[shoulder > LIGHT_SHOULDER_THRESHOLD] = AnalogShoulder.LIGHT  # light shield
  buckets[shoulder > FULL_SHOULDER_THRESHOLD] = AnalogShoulder.FULL  # full press
  return buckets

# 0.35 is the value from pressing Z
SHOULDER_TABLE = np.array([0, 0.35, 1], dtype=np.float32)

def decode_shoulder(bucket: UInt8Array[S]) -> FloatArray[S]:
  return tp.cast(FloatArray[S], SHOULDER_TABLE[bucket])

IntArray = BoolArray[S] | UInt8Array[S] | UInt16Array[S]

class Cartesian:

  def __init__(self, axis_specs: tp.Sequence[tuple[int, type]]):
    self.axis_specs = axis_specs
    self.axis_sizes = [size for size, _ in axis_specs]
    self.num_labels = np.prod(self.axis_sizes)

  def flatten(self, components: tp.Sequence[IntArray[S]]) -> UInt16Array[S]:
    shape = components[0].shape
    label = np.zeros(shape, dtype=np.uint16)

    for size, component in zip(self.axis_sizes, components):
      assert np.all(component < size)

      label *= np.uint16(size)
      label += component.astype(np.uint16)

    return tp.cast(UInt16Array[S], label)

  def unflatten(self, label: UInt16Array[S]) -> list[UInt16Array[S]]:
    components = []
    for size, spec in reversed(self.axis_specs):
      label, component = np.divmod(label, size)  # type: ignore
      components.append(component.astype(spec))

    components.reverse()
    return components

a_and_shoulder_bucketer = Cartesian(axis_specs=[(2, bool), (len(AnalogShoulder), np.uint8)])
A_AND_SHOULDER_SIZE = a_and_shoulder_bucketer.num_labels
Z_LABEL = np.uint8(a_and_shoulder_bucketer.num_labels)

def bucket_z_a_shoulder(controller: Controller[S]) -> UInt16Array[S]:
  """Bucket Z, A, and shoulder into a single feature with 7 buckets."""
  z = controller.buttons.Z
  a = controller.buttons.A
  shoulder_bucket = bucket_analog_shoulder(controller.shoulder)

  a_and_shoulder = a_and_shoulder_bucketer.flatten((a, shoulder_bucket))

  combined = a_and_shoulder
  combined[z] = Z_LABEL
  return combined

def decode_z_a_shoulder(bucket: UInt16Array[S]) -> tuple[BoolArray[S], BoolArray[S], FloatArray[S]]:
  """Decode the combined Z/A/shoulder bucket back into its components."""
  z = bucket == Z_LABEL

  no_z = bucket[~z]

  no_z_a, no_z_shoulder = a_and_shoulder_bucketer.unflatten(no_z)

  a = np.zeros_like(bucket, dtype=bool)
  a[~z] = no_z_a.astype(bool)
  a[z] = True
  a = tp.cast(BoolArray[S], a)

  shoulder_label = np.zeros_like(bucket, dtype=np.uint8)
  shoulder_label[~z] = no_z_shoulder.astype(np.uint8)
  shoulder_label[z] = AnalogShoulder.LIGHT

  shoulder = decode_shoulder(shoulder_label)
  shoulder = tp.cast(FloatArray[S], shoulder)

  return z, a, shoulder

Z_A_SHOULDER_SIZE = A_AND_SHOULDER_SIZE + 1

def stick_to_raw(value: FloatArray[S]) -> np.ndarray[S, np.dtype[np.int16]]:
  """Convert a normalized float array in [0, 1] to raw controller values."""
  return np.rint(value * 160 - 80).astype(np.int16)  # type: ignore

def stick_from_raw(value: np.ndarray[S, tp.Any]) -> FloatArray[S]:
  """Convert raw controller values to a normalized float array in [0, 1]."""
  return ((value + 80) / 160).astype(np.float32)  # type: ignore

# In raw coordinates
MIN_NONZERO_RADIUS = 23
MAX_RADIUS = 80

min_log_radius = np.log(MIN_NONZERO_RADIUS)
max_log_radius = np.log(MAX_RADIUS)

def build_radius_table(n_radius_buckets: int) -> FloatArray[tuple[int]]:
  """Get the radius value of each radius bucket."""
  nonzero_radii = np.exp(np.linspace(min_log_radius, max_log_radius, n_radius_buckets))
  return np.concatenate(([0.0], nonzero_radii)).astype(np.float32)

def build_angle_table(n_angle_buckets: int) -> FloatArray[tuple[int]]:
  """Get the angle value of each angle bucket."""
  normalized_angles = np.arange(n_angle_buckets) / n_angle_buckets
  # Map to [-pi, pi)
  return (normalized_angles * 2 * np.pi - np.pi).astype(np.float32)

class RaggedCartesian:

  def __init__(self, bucket_sizes: tp.Sequence[int]):
    self.bucket_sizes = bucket_sizes
    self.radius_label_offset = np.cumsum([0, *bucket_sizes[:-1]])
    self.num_labels = sum(bucket_sizes)

    self.label_to_outer = np.concatenate([
        np.full(size, i, dtype=np.uint16)
        for i, size in enumerate(bucket_sizes)])
    self.label_to_inner = np.concatenate([
        np.arange(size, dtype=np.uint16)
        for size in bucket_sizes])

  def to_label(self, outer_index: UInt16Array[S], inner_index: UInt16Array[S]) -> UInt16Array[S]:
    labels = self.radius_label_offset[outer_index] + inner_index
    return tp.cast(UInt16Array[S], labels)

  def from_label(self, label: UInt16Array[S]) -> tuple[UInt16Array[S], UInt16Array[S]]:
    outer = tp.cast(UInt16Array[S], self.label_to_outer[label])
    inner = tp.cast(UInt16Array[S], self.label_to_inner[label])
    return outer, inner

class PolarStickBucketer:

  def __init__(self, n_radius_buckets: int, n_angle_buckets: int | tp.Sequence[int]):
    self.n_radius_buckets = n_radius_buckets
    if isinstance(n_angle_buckets, int):
      self.n_angle_buckets = [n_angle_buckets] * n_radius_buckets
    else:
      if len(n_angle_buckets) != n_radius_buckets:
        raise ValueError(f"n_angle_buckets ({n_angle_buckets}) must match n_radius_buckets ({n_radius_buckets})")
      self.n_angle_buckets = n_angle_buckets

    # Maps radius bucket (including 0) back to (raw) radius value
    self.radius_table = build_radius_table(n_radius_buckets)

    # Maps radius bucket to number of angle buckets for that radius
    self.n_angle_buckets_table = np.array([1, *self.n_angle_buckets])

    # Maps flat label back to angle in radians
    self.angle_table = np.concatenate([
        build_angle_table(n) for n in self.n_angle_buckets_table])

    # Maps (radius bucket, angle bucket) to flat label
    self.ragged_cartesian = RaggedCartesian(self.n_angle_buckets_table)
    self.num_labels = self.ragged_cartesian.num_labels

    assert self.angle_table.shape == (self.num_labels,)

  def cartesian_to_polar(
      self, stick: Stick[S],
  ) -> tuple[UInt16Array[S], UInt16Array[S]]:
    # Convert to raw [-80, 80] coordinates
    raw_x = stick_to_raw(stick.x)
    raw_y = stick_to_raw(stick.y)
    radius = np.sqrt(raw_x**2 + raw_y**2)
    is_origin = radius <= (MIN_NONZERO_RADIUS - 1)

    normalized_radius = (np.log(radius + 1e-3) - min_log_radius) / (max_log_radius - min_log_radius)
    nonzero_radius_bucket = np.rint(normalized_radius * (self.n_radius_buckets - 1)).astype(np.uint16)

    # Assign the origin bucket (0) to all sticks within the deadzone, and then
    # assign non-origin buckets starting from 1.
    radius_bucket = np.where(is_origin, 0, nonzero_radius_bucket + 1)
    radius_bucket = tp.cast(UInt16Array[S], radius_bucket)

    # The number of angle buckets for each datapoint
    n_angle_buckets = self.n_angle_buckets_table[radius_bucket]
    angle = np.arctan2(raw_y, raw_x)  # [-pi, pi]
    normalized_angle = (angle + np.pi) / (2 * np.pi)  # [0, 1]
    angle_bucket = np.rint(normalized_angle * n_angle_buckets).astype(np.uint16)
    # Wrap around the angle bucket to ensure it falls within [0, n_angle_buckets - 1]
    angle_bucket = np.mod(angle_bucket, n_angle_buckets)
    angle_bucket = tp.cast(UInt16Array[S], angle_bucket)
    return radius_bucket, angle_bucket

  def bucket(self, stick: Stick[S]) -> UInt16Array[S]:
    radius_bucket, angle_bucket = self.cartesian_to_polar(stick)
    return self.ragged_cartesian.to_label(radius_bucket, angle_bucket)

  def decode(self, label: UInt16Array[S]) -> Stick[S]:
    radius_bucket, _ = self.ragged_cartesian.from_label(label)

    raw_radius = self.radius_table[radius_bucket]
    angle = self.angle_table[label]

    raw_x = raw_radius * np.cos(angle)
    raw_y = raw_radius * np.sin(angle)

    x = stick_from_raw(raw_x)
    y = stick_from_raw(raw_y)

    return Stick(x, y)

@dataclasses.dataclass
class PolarStickConfig:
  n_radius_buckets: int
  n_angle_buckets: tp.Sequence[int]

  def create_bucketer(self) -> PolarStickBucketer:
    return PolarStickBucketer(self.n_radius_buckets, self.n_angle_buckets)

class ButtonCombination(tp.NamedTuple, tp.Generic[S]):
  b: BoolArray[S]
  xy: BoolArray[S]
  lr: BoolArray[S]
  z_a_shoulder: UInt16Array[S]  # Output of bucket_z_a_shoulder, with size Z_A_SHOULDER_SIZE

  # TODO: c-stick is generally not independent of the buttons, we might want to
  # combine it with the buttons instead of using a cartesian product.
  c_stick: UInt16Array[S]  # PolarStickBucketer label

class CombinedButtonBucketer:

  def __init__(self, c_stick_config: PolarStickConfig):
    self.c_stick_bucketer = c_stick_config.create_bucketer()

    _bool = (2, np.bool)
    axis_specs = ButtonCombination(
        b=_bool, xy=_bool, lr=_bool,  # type: ignore
        z_a_shoulder=(Z_A_SHOULDER_SIZE, np.uint8),  # type: ignore
        c_stick=(self.c_stick_bucketer.num_labels, np.uint16),  # type: ignore
    )

    self.cartesian = Cartesian(axis_specs)  # type: ignore
    self.num_labels = self.cartesian.num_labels

  def controller_to_button_combination(self, controller: Controller[S]) -> ButtonCombination[S]:
    lr = tp.cast(BoolArray[S], controller.buttons.L | controller.buttons.R)
    c_stick_bucket = self.c_stick_bucketer.bucket(controller.c_stick)
    xy = tp.cast(BoolArray[S], controller.buttons.X | controller.buttons.Y)
    z_a_shoulder = bucket_z_a_shoulder(controller)

    return ButtonCombination(
        b=controller.buttons.B,
        xy=xy,
        lr=lr,
        z_a_shoulder=z_a_shoulder,
        c_stick=c_stick_bucket,
    )

  def bucket(self, controller: Controller[S]) -> UInt16Array[S]:
    combo = self.controller_to_button_combination(controller)
    return self.cartesian.flatten(combo)

  def from_label(self, label: UInt16Array[S]) -> tuple[ButtonCombination[S], Stick[S]]:
    components = self.cartesian.unflatten(label)
    button_combination = ButtonCombination(*components)
    c_stick = self.c_stick_bucketer.decode(button_combination.c_stick)
    return button_combination, c_stick


class ControllerBucketer:

  def __init__(self, c_stick_config: PolarStickConfig, main_stick_config: PolarStickConfig):
    self.button_bucketer = CombinedButtonBucketer(c_stick_config)
    self.main_stick_bucketer = main_stick_config.create_bucketer()
    self.axis_sizes = (self.button_bucketer.num_labels, self.main_stick_bucketer.num_labels)

  def bucket(self, controller: Controller[S]) -> tuple[UInt16Array[S], UInt16Array[S]]:
    button_label = self.button_bucketer.bucket(controller)
    main_stick_label = self.main_stick_bucketer.bucket(controller.main_stick)
    return button_label, main_stick_label

  def decode(self, button_label: UInt16Array[S], main_stick_label: UInt16Array[S]) -> Controller[S]:
    button_combination, c_stick = self.button_bucketer.from_label(button_label)
    main_stick = self.main_stick_bucketer.decode(main_stick_label)

    z, a, shoulder = decode_z_a_shoulder(button_combination.z_a_shoulder)

    return Controller(
        main_stick=main_stick,
        c_stick=c_stick,
        shoulder=shoulder,
        buttons=Buttons(
            A=a,
            B=button_combination.b,
            X=np.zeros_like(button_combination.xy),
            Y=button_combination.xy,
            Z=z,
            L=button_combination.lr,
            R=np.zeros_like(button_combination.lr),
            D_UP=np.zeros_like(button_combination.b, dtype=bool),
        )
    )

@dataclasses.dataclass
class Config:
  c_stick_config: PolarStickConfig
  main_stick_config: PolarStickConfig

  @classmethod
  def default(cls) -> tp.Self:
    return cls(
        c_stick_config=PolarStickConfig(n_radius_buckets=2, n_angle_buckets=[4, 8]),
        main_stick_config=PolarStickConfig(n_radius_buckets=3, n_angle_buckets=[4, 16, 64]),
    )

  def create_bucketer(self) -> ControllerBucketer:
    return ControllerBucketer(self.c_stick_config, self.main_stick_config)
