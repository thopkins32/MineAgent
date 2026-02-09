package com.mineagent;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class ObservationTest {

  @Test
  void serialize_returnsFrameByteArray() {
    byte[] frame = {1, 2, 3, 4, 5};
    Observation obs = new Observation(10.5, frame);

    byte[] result = obs.serialize();

    assertArrayEquals(frame, result);
    assertSame(frame, result); // Should return the same reference
  }

  @Test
  void serialize_withEmptyFrame_returnsEmptyArray() {
    byte[] frame = {};
    Observation obs = new Observation(0.0, frame);

    byte[] result = obs.serialize();

    assertArrayEquals(frame, result);
    assertEquals(0, result.length);
  }

  @Test
  void serialize_withLargeFrame_returnsCorrectArray() {
    byte[] frame = new byte[1024 * 1024]; // 1MB frame
    for (int i = 0; i < frame.length; i++) {
      frame[i] = (byte) (i % 256);
    }
    Observation obs = new Observation(100.0, frame);

    byte[] result = obs.serialize();

    assertArrayEquals(frame, result);
    assertEquals(frame.length, result.length);
  }

  @Test
  void recordEquality_withSameValues_areEqual() {
    byte[] frame1 = {1, 2, 3};
    byte[] frame2 = {1, 2, 3};
    Observation obs1 = new Observation(5.0, frame1);
    Observation obs2 = new Observation(5.0, frame2);

    assertEquals(obs1, obs2);
    assertEquals(obs1.hashCode(), obs2.hashCode());
  }

  @Test
  void recordEquality_withDifferentReward_areNotEqual() {
    byte[] frame = {1, 2, 3};
    Observation obs1 = new Observation(5.0, frame);
    Observation obs2 = new Observation(10.0, frame);

    assertNotEquals(obs1, obs2);
  }

  @Test
  void recordEquality_withDifferentFrame_areNotEqual() {
    byte[] frame1 = {1, 2, 3};
    byte[] frame2 = {1, 2, 4};
    Observation obs1 = new Observation(5.0, frame1);
    Observation obs2 = new Observation(5.0, frame2);

    assertNotEquals(obs1, obs2);
  }

  @Test
  void recordEquality_withSameReference_areEqual() {
    byte[] frame = {1, 2, 3};
    Observation obs1 = new Observation(5.0, frame);
    Observation obs2 = new Observation(5.0, frame);

    assertEquals(obs1, obs2);
  }

  @Test
  void constructor_withVariousRewardValues_storesCorrectly() {
    Observation obs1 = new Observation(0.0, new byte[] {1});
    Observation obs2 = new Observation(-100.0, new byte[] {1});
    Observation obs3 = new Observation(100.0, new byte[] {1});
    Observation obs4 = new Observation(Double.MAX_VALUE, new byte[] {1});
    Observation obs5 = new Observation(Double.MIN_VALUE, new byte[] {1});

    assertEquals(0.0, obs1.reward());
    assertEquals(-100.0, obs2.reward());
    assertEquals(100.0, obs3.reward());
    assertEquals(Double.MAX_VALUE, obs4.reward());
    assertEquals(Double.MIN_VALUE, obs5.reward());
  }

  @Test
  void constructor_withVariousFrameSizes_storesCorrectly() {
    Observation obs1 = new Observation(0.0, new byte[0]);
    Observation obs2 = new Observation(0.0, new byte[1]);
    Observation obs3 = new Observation(0.0, new byte[1000]);
    Observation obs4 = new Observation(0.0, new byte[1000000]);

    assertEquals(0, obs1.frame().length);
    assertEquals(1, obs2.frame().length);
    assertEquals(1000, obs3.frame().length);
    assertEquals(1000000, obs4.frame().length);
  }

  @Test
  void toString_includesRewardAndFrameInfo() {
    byte[] frame = {1, 2, 3};
    Observation obs = new Observation(42.5, frame);

    String str = obs.toString();

    assertNotNull(str);
    assertTrue(str.contains("42.5") || str.contains("42"));
    // Record toString typically includes class name and field values
  }
}
