import time
import cv2
import mediapipe as mp

# --- Setup: hand tracker (lightweight model) ---
hands = mp.solutions.hands.Hands(
    max_num_hands=2,             # track both hands
    model_complexity=0,          # 0 = fastest, good for regular laptops
    min_detection_confidence=0.4,   # lowered: locks onto your hand faster
    min_tracking_confidence=0.4,    # lowered: keeps tracking through quick motion
)

# --- Setup: webcam ---
cap = cv2.VideoCapture(0)        # 0 = default webcam; try 1 for a USB cam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- Game settings ---
BALL_RADIUS = 30
HAND_RADIUS = 50
GRAVITY = 0.6              # how fast the ball accelerates downward
BOUNCE_STRENGTH = 18       # overall speed applied when it bounces off your hand
SIDEWAYS_INFLUENCE = 0.9   # how much your hand's sideways motion pushes the ball
MAX_BALL_SPEED = 22         # caps how fast the ball can ever move, so it never flies wildly

WINDOW_NAME = "Basketball Bounce  (Q to quit)"
COUNTDOWN_SECONDS = 3


def reset_ball(w):
    """Ball starts near the top-middle, falling slowly."""
    return {"x": w // 2, "y": 50, "vx": 0.0, "vy": 2.0}


ball = None
score = 0
game_over = False
countdown_end = time.time() + COUNTDOWN_SECONDS   # brief pause before play begins

# Track each hand's previous position so we can work out how fast
# and in what direction it is moving (needed for a natural-feeling bounce).
prev_hand_positions = {}
prev_hand_velocities = {}

# The window itself can be freely resized/stretched by the user, but we
# manually letterbox the picture before showing it, so the camera feed
# always keeps its original proportions with black bars filling any
# leftover space, instead of stretching.
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 640, 480)


def fill_window(image, target_w, target_h):
    """Resize `image` to completely fill (target_w, target_h) without
    distorting it: scales up to cover the whole area, then crops off
    whatever spills over the edges (no black bars, no stretching)."""
    src_h, src_w = image.shape[:2]
    if target_w <= 0 or target_h <= 0:
        return image

    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
    resized = cv2.resize(image, (new_w, new_h))

    x_start = (new_w - target_w) // 2
    y_start = (new_h - target_h) // 2
    cropped = resized[y_start:y_start + target_h, x_start:x_start + target_w]
    return cropped

while True:
    ok, frame = cap.read()
    if not ok:
        print("Could not read from webcam.")
        break

    # Not mirrored: the frame is shown exactly as the camera sees it.
    h, w, _ = frame.shape
    if ball is None:
        ball = reset_ball(w)

    # --- Find the hands ---
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_positions = []
    hand_velocities = []
    if result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # Landmark 9 = base of the middle finger (roughly the palm centre)
            lm = hand_landmarks.landmark[9]
            pos = (int(lm.x * w), int(lm.y * h))
            hand_positions.append(pos)
            cv2.circle(frame, pos, HAND_RADIUS, (0, 255, 0), 3)

            # Work out how fast this hand moved since the last frame,
            # smoothed against the previous velocity so a single jittery
            # frame from the tracker doesn't register as a fast swipe.
            prev = prev_hand_positions.get(i)
            if prev is not None:
                raw_vx = pos[0] - prev[0]
                raw_vy = pos[1] - prev[1]
            else:
                raw_vx, raw_vy = 0.0, 0.0
            prev_vx, prev_vy = prev_hand_velocities.get(i, (0.0, 0.0))
            vx = prev_vx * 0.5 + raw_vx * 0.5
            vy = prev_vy * 0.5 + raw_vy * 0.5
            hand_velocities.append((vx, vy))
            prev_hand_positions[i] = pos
            prev_hand_velocities[i] = (vx, vy)

    # Clear stale tracked hands if fewer hands are visible this frame.
    for key in list(prev_hand_positions.keys()):
        if key >= len(hand_positions):
            del prev_hand_positions[key]
    for key in list(prev_hand_velocities.keys()):
        if key >= len(hand_positions):
            del prev_hand_velocities[key]

    in_countdown = countdown_end is not None and time.time() < countdown_end

    if not game_over and not in_countdown:
        # --- Physics: gravity pulls the ball down ---
        ball["vy"] += GRAVITY
        ball["x"] += ball["vx"]
        ball["y"] += ball["vy"]

        # Slight air friction so sideways speed doesn't carry on forever.
        ball["vx"] *= 0.98

        # --- Check for a bounce off either hand ---
        for hand_pos, hand_vel in zip(hand_positions, hand_velocities):
            dx = ball["x"] - hand_pos[0]
            dy = ball["y"] - hand_pos[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < BALL_RADIUS + HAND_RADIUS and ball["vy"] > 0:
                hvx, hvy = hand_vel
                # No snapping to the hand's centre: the ball keeps its own
                # position and simply has its velocity redirected, so the
                # bounce angle follows wherever on the circle it was touched
                # plus how fast your hand was swiping when it connected.
                ball["vy"] = -(BOUNCE_STRENGTH + abs(hvy) * 0.4)
                ball["vx"] = ball["vx"] * 0.3 + hvx * SIDEWAYS_INFLUENCE
                # Clamp so a fast or jittery swipe never sends the ball flying uncontrollably.
                ball["vx"] = max(-MAX_BALL_SPEED, min(MAX_BALL_SPEED, ball["vx"]))
                ball["vy"] = max(-MAX_BALL_SPEED, min(MAX_BALL_SPEED, ball["vy"]))
                score += 1
                break

        # --- Bounce off the left/right edges instead of clipping through ---
        if ball["x"] - BALL_RADIUS < 0:
            ball["x"] = BALL_RADIUS
            ball["vx"] *= -0.6
        elif ball["x"] + BALL_RADIUS > w:
            ball["x"] = w - BALL_RADIUS
            ball["vx"] *= -0.6

        # --- Missed: ball fell past the bottom ---
        if ball["y"] - BALL_RADIUS > h:
            game_over = True

    # --- Draw the ball ---
    cv2.circle(frame, (int(ball["x"]), int(ball["y"])), BALL_RADIUS, (0, 140, 255), -1)

    # Match the frame to whatever size the user has resized the window to.
    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
    except cv2.error:
        win_w, win_h = w, h
    display_frame = fill_window(frame, win_w, win_h) if win_w > 0 and win_h > 0 else frame
    dh, dw = display_frame.shape[:2]

    # Text is drawn AFTER cropping to the display size, so it always stays
    # anchored to the visible corners of the window instead of possibly
    # landing in the part of the image that got cropped away.
    if in_countdown:
        seconds_left = int(countdown_end - time.time()) + 1
        cv2.putText(display_frame, f"Get ready: {seconds_left}", (dw // 2 - 150, dh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        cv2.putText(display_frame, f"Score: {score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    elif not game_over:
        cv2.putText(display_frame, f"Score: {score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    else:
        cv2.putText(display_frame, f"GAME OVER - Score: {score}", (dw // 2 - 220, dh // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(display_frame, "Press R to Restart or Q to Quit", (dw // 2 - 260, dh // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r") and game_over:
        ball = reset_ball(w)
        score = 0
        game_over = False
        countdown_end = time.time() + COUNTDOWN_SECONDS

cap.release()
cv2.destroyAllWindows()