package com.mvi.mvimod;

import java.nio.ByteBuffer;

/*
 * This class represents the schema for communication between the client and server.
 * It is a data structure that can pack and unpack the action state
 * into a byte array and back.
 * 
 * The format is as follows:
 * - UP (1 bit)
 * - DOWN (1 bit)
 * - LEFT (1 bit)
 * - RIGHT (1 bit)
 * - JUMP (1 bit)
 * - SNEAK (1 bit)
 * - SPRINT (1 bit)
 * - INVENTORY (1 bit)
 * - DROP (1 bit)
 * - SWAP (1 bit)
 * - USE (1 bit)
 * - ATTACK (1 bit)
 * - PICK_ITEM (1 bit)
 * - HOTBAR_1 (1 bit)
 * - HOTBAR_2 (1 bit)
 * - HOTBAR_3 (1 bit)
 * - HOTBAR_4 (1 bit)
 * - HOTBAR_5 (1 bit)
 * - HOTBAR_6 (1 bit)
 * - HOTBAR_7 (1 bit)
 * - HOTBAR_8 (1 bit)
 * - EXIT_MENU (1 bit)
 * - PADDING (1 bit)
 * - PADDING (1 bit)
 * - Mouse control X (4 bytes)
 * - Mouse control Y (4 bytes)
 */
public record Action(
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
    boolean hotbar8,
    boolean exitMenu,
    float mouseControlX,
    float mouseControlY
) {
    public byte[] toBytes() {
        ByteBuffer buffer = ByteBuffer.allocate(3 + 4 + 4);

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
        thirdByte <<= 1;
        thirdByte |= (exitMenu ? 1 : 0);
        buffer.put((byte) thirdByte);

        buffer.putFloat(mouseControlX);
        buffer.putFloat(mouseControlY);

        return buffer.array();
    }

    public static Action fromBytes(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes);

        int firstByte = buffer.get();
        int secondByte = buffer.get();
        int thirdByte = buffer.get();

        float mouseControlX = buffer.getFloat();
        float mouseControlY = buffer.getFloat();

        return new Action(
            ((firstByte >> 0) & 1) == 1,
            ((firstByte >> 1) & 1) == 1,
            ((firstByte >> 2) & 1) == 1,
            ((firstByte >> 3) & 1) == 1,
            ((firstByte >> 4) & 1) == 1,
            ((firstByte >> 5) & 1) == 1,
            ((firstByte >> 6) & 1) == 1,
            ((firstByte >> 7) & 1) == 1,
            ((secondByte >> 0) & 1) == 1,
            ((secondByte >> 1) & 1) == 1,
            ((secondByte >> 2) & 1) == 1,
            ((secondByte >> 3) & 1) == 1,
            ((secondByte >> 4) & 1) == 1,
            ((secondByte >> 5) & 1) == 1,
            ((secondByte >> 6) & 1) == 1,
            ((secondByte >> 7) & 1) == 1,
            ((thirdByte >> 0) & 1) == 1,
            ((thirdByte >> 1) & 1) == 1,
            ((thirdByte >> 2) & 1) == 1,
            ((thirdByte >> 3) & 1) == 1,
            ((thirdByte >> 4) & 1) == 1,
            ((thirdByte >> 5) & 1) == 1,
            mouseControlX,
            mouseControlY
        );
    }
}
