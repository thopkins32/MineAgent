package com.mvi.mvimod;

import java.nio.ByteBuffer;

public record ActionState(
    boolean up,
    boolean down,
    boolean left,
    boolean right,
    boolean jump,
    boolean sneak,
    boolean sprint,
    boolean inventory,
    boolean drop,
    boolean swap,
    boolean use,
    boolean attack,
    boolean pick_item,
    boolean hotbar1,
    boolean hotbar2,
    boolean hotbar3,
    boolean hotbar4,
    boolean hotbar5,
    boolean hotbar6,
    boolean hotbar7,
    boolean hotbar8
) {
    public byte[] toBytes() {
        ByteBuffer buffer = ByteBuffer.allocate(3);

        int firstByte = 0;
        firstByte |= (up ? 1 : 0);
        firstByte <<= 1;
        firstByte |= (down ? 1 : 0);
        firstByte <<= 1;
        firstByte |= (left ? 1 : 0);
        firstByte <<= 1;
        firstByte |= (right ? 1 : 0);
        firstByte <<= 1;
        firstByte |= (jump ? 1 : 0);
        firstByte <<= 1;
        firstByte |= (sneak ? 1 : 0);
        firstByte <<= 1;
        firstByte |= (sprint ? 1 : 0);
        firstByte <<= 1;
        firstByte |= (inventory ? 1 : 0);
        buffer.put((byte) firstByte);

        int secondByte = 0;
        secondByte |= (drop ? 1 : 0);
        secondByte <<= 1;
        secondByte |= (swap ? 1 : 0);
        secondByte <<= 1;
        secondByte |= (use ? 1 : 0);
        secondByte <<= 1;
        secondByte |= (attack ? 1 : 0);
        secondByte <<= 1;
        secondByte |= (pick_item ? 1 : 0);
        secondByte <<= 1;
        secondByte |= (hotbar1 ? 1 : 0);
        secondByte <<= 1;
        secondByte |= (hotbar2 ? 1 : 0);
        secondByte <<= 1;
        secondByte |= (hotbar3 ? 1 : 0);
        buffer.put((byte) secondByte);
        
        int thirdByte = 0;
        thirdByte |= (hotbar4 ? 1 : 0);
        thirdByte <<= 1;
        thirdByte |= (hotbar5 ? 1 : 0);
        thirdByte <<= 1;
        thirdByte |= (hotbar6 ? 1 : 0);
        thirdByte <<= 1;
        thirdByte |= (hotbar7 ? 1 : 0);
        thirdByte <<= 1;
        thirdByte |= (hotbar8 ? 1 : 0);
        // Leave 3 bits unused for padding
        buffer.put((byte) thirdByte);

        return buffer.array();
    }

    public static ActionState fromBytes(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes);

        int firstByte = buffer.get() & 0xFF;
        int secondByte = buffer.get() & 0xFF;
        int thirdByte = buffer.get() & 0xFF;

        return new ActionState(
            ((firstByte >> 7) & 1) == 1,  // up
            ((firstByte >> 6) & 1) == 1,  // down
            ((firstByte >> 5) & 1) == 1,  // left
            ((firstByte >> 4) & 1) == 1,  // right
            ((firstByte >> 3) & 1) == 1,  // jump
            ((firstByte >> 2) & 1) == 1,  // sneak
            ((firstByte >> 1) & 1) == 1,  // sprint
            ((firstByte >> 0) & 1) == 1,  // inventory
            ((secondByte >> 7) & 1) == 1, // drop
            ((secondByte >> 6) & 1) == 1, // swap
            ((secondByte >> 5) & 1) == 1, // use
            ((secondByte >> 4) & 1) == 1, // attack
            ((secondByte >> 3) & 1) == 1, // pick_item
            ((secondByte >> 2) & 1) == 1, // hotbar1
            ((secondByte >> 1) & 1) == 1, // hotbar2
            ((secondByte >> 0) & 1) == 1, // hotbar3
            ((thirdByte >> 7) & 1) == 1,  // hotbar4
            ((thirdByte >> 6) & 1) == 1,  // hotbar5
            ((thirdByte >> 5) & 1) == 1,  // hotbar6
            ((thirdByte >> 4) & 1) == 1,  // hotbar7
            ((thirdByte >> 3) & 1) == 1   // hotbar8
        );
    }
}
