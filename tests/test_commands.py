"""Tests for vendor command packing logic.

The T2 Pro sends vendor commands via V4L2_CID_ZOOM_ABSOLUTE.
Commands are packed as 16-bit integers: (command_byte << 8) | data_byte.
"""

import pytest
from infiray_t2pro.commands import Command, pack_command


class TestCommandEnum:
    def test_shutter_close_value(self):
        assert Command.SHUTTER_CLOSE == 0x8000

    def test_high_gain_value(self):
        assert Command.HIGH_GAIN == 0x8020

    def test_low_gain_value(self):
        assert Command.LOW_GAIN == 0x8021

    def test_palette_white_hot_value(self):
        assert Command.PALETTE_WHITE_HOT == 0x8800

    def test_palette_iron_value(self):
        assert Command.PALETTE_IRON == 0x8802

    def test_default_value(self):
        assert Command.DEFAULT == 1


class TestPackCommand:
    def test_packs_command_with_zero_data(self):
        result = pack_command(0x80, 0x00)
        assert result == 0x8000

    def test_packs_command_with_data_byte(self):
        result = pack_command(0x80, 0xFF)
        assert result == 0x80FF

    def test_packs_command_with_mid_range_data(self):
        result = pack_command(0x80, 0x05)
        assert result == 0x8005

    def test_data_byte_masked_to_8_bits(self):
        """Data larger than a byte should be masked to 0xFF."""
        result = pack_command(0x80, 0x1FF)
        assert result == 0x80FF

    def test_temperature_param_position_0(self):
        """Temperature params are sent as (position * 4, byte_value)."""
        result = pack_command(0 * 4, 0x05)
        assert result == 0x0005

    def test_temperature_param_position_1(self):
        result = pack_command(1 * 4, 0x0A)
        assert result == 0x040A

    def test_temperature_param_position_5(self):
        result = pack_command(5 * 4, 0xFF)
        assert result == 0x14FF

    def test_command_value_from_enum(self):
        """pack_command should accept Command enum values as the command byte."""
        result = pack_command(0x80, 0)
        assert result == int(Command.SHUTTER_CLOSE)