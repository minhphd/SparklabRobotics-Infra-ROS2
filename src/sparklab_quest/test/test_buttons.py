from sparklab_quest import buttons


def test_short_arrays_read_as_unpressed():
    assert not buttons.pressed([], buttons.GRIP)
    assert buttons.value([], buttons.TRIGGER) == 0.0


def test_pressed_and_value_index_the_joy_arrays():
    assert buttons.pressed([1, 0], buttons.TRIGGER)
    assert not buttons.pressed([1, 0], buttons.GRIP)
    assert buttons.value([0.75], buttons.TRIGGER) == 0.75
