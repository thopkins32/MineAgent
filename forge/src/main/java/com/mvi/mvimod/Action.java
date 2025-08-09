package com.mvi.mvimod;

import java.nio.ByteBuffer;
import com.mojang.logging.LogUtils;
import org.slf4j.Logger;

/*
 * This class represents the complete action schema for communication between the client and server.
 * It contains both persistent actions (button states that continue until released) and 
 * non-persistent actions (like mouse movement and menu commands).
 * 
 * The format is as follows:
 * - Persistent actions (3 bytes) - see ActionState for bit layout
 * - EXIT_MENU (1 bit)
 * - PADDING (7 bits)
 * - Mouse control X (4 bytes)
 * - Mouse control Y (4 bytes)
 */
public record Action(
    ActionState actionState,
    boolean exitMenu,
    float mouseControlX,
    float mouseControlY
) {
    private static final Logger LOGGER = LogUtils.getLogger();

    public boolean up() { return actionState.up(); }
    public boolean down() { return actionState.down(); }
    public boolean left() { return actionState.left(); }
    public boolean right() { return actionState.right(); }
    public boolean jump() { return actionState.jump(); }
    public boolean sneak() { return actionState.sneak(); }
    public boolean sprint() { return actionState.sprint(); }
    public boolean inventory() { return actionState.inventory(); }
    public boolean drop() { return actionState.drop(); }
    public boolean swap() { return actionState.swap(); }
    public boolean use() { return actionState.use(); }
    public boolean attack() { return actionState.attack(); }
    public boolean pick_item() { return actionState.pick_item(); }
    public boolean hotbar1() { return actionState.hotbar1(); }
    public boolean hotbar2() { return actionState.hotbar2(); }
    public boolean hotbar3() { return actionState.hotbar3(); }
    public boolean hotbar4() { return actionState.hotbar4(); }
    public boolean hotbar5() { return actionState.hotbar5(); }
    public boolean hotbar6() { return actionState.hotbar6(); }
    public boolean hotbar7() { return actionState.hotbar7(); }
    public boolean hotbar8() { return actionState.hotbar8(); }
    public boolean rightMouseDown() { return actionState.rightMouseDown(); }
    public boolean leftMouseDown() { return actionState.leftMouseDown(); }

    public byte[] toBytes() {
        ByteBuffer buffer = ByteBuffer.allocate(3 + 1 + 4 + 4);

        // Add persistent action bytes
        byte[] persistentBytes = actionState.toBytes();
        buffer.put(persistentBytes);

        // Add exit menu flag with padding
        int menuByte = 0;
        menuByte |= (exitMenu ? 1 << 7 : 0);
        buffer.put((byte) menuByte);

        // Add mouse controls
        buffer.putFloat(mouseControlX);
        buffer.putFloat(mouseControlY);

        return buffer.array();
    }

    public static Action fromBytes(byte[] bytes) {
        LOGGER.info("Action fromBytes: {}", bytes);
        ByteBuffer buffer = ByteBuffer.wrap(bytes);

        // Extract persistent action bytes
        byte[] persistentBytes = new byte[3];
        buffer.get(persistentBytes);
        ActionState actionState = ActionState.fromBytes(persistentBytes);

        // Extract exit menu flag
        int menuByte = buffer.get() & 0xFF;
        boolean exitMenu = ((menuByte >> 7) & 1) == 1;

        // Extract mouse controls
        float mouseControlX = buffer.getFloat();
        float mouseControlY = buffer.getFloat();

        return new Action(actionState, exitMenu, mouseControlX, mouseControlY);
    }

    // Convenience constructor for creating actions with individual persistent fields
    public Action(
        boolean up, boolean down, boolean left, boolean right,
        boolean jump, boolean sneak, boolean sprint, boolean inventory,
        boolean drop, boolean swap, boolean use, boolean attack, boolean pick_item,
        boolean hotbar1, boolean hotbar2, boolean hotbar3, boolean hotbar4,
        boolean hotbar5, boolean hotbar6, boolean hotbar7, boolean hotbar8,
        boolean rightMouseDown, boolean leftMouseDown,
        boolean exitMenu, float mouseControlX, float mouseControlY
    ) {
        this(
            new ActionState(
                up, down, left, right, jump, sneak, sprint, inventory,
                drop, swap, use, attack, pick_item,
                hotbar1, hotbar2, hotbar3, hotbar4, hotbar5, hotbar6, hotbar7, hotbar8,
                rightMouseDown, leftMouseDown
            ),
            exitMenu, mouseControlX, mouseControlY
        );
    }
} 