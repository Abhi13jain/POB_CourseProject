import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np


#192x192 model
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']


def draw_keypoints_on_image(image_bgr, keypoints_single, confidence_threshold=0.3):
    # keypoints_single: numpy array shape [17, 3] (y, x, score)
    output = image_bgr.copy()
    h, w = output.shape[:2]
    for (y, x, kp_score) in keypoints_single:
        if kp_score < confidence_threshold:
            continue
        cx = int(x * w)
        cy = int(y * h)
        cv2.circle(output, (cx, cy), 3, (0, 255, 0), -1)
    return output


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0)")

    # Calibration and FPS state
    baseline_distance = None
    posture_status = "Unknown"
    bad_posture_frames = 0
    good_posture_frames = 0
    import time
    last_time = time.time()
    frame_count = 0
    displayed_fps = 0.0
    target_size = 192

    # Warmup to JIT/initialize graph and improve steady-state FPS
    warm = tf.ones((1, target_size, target_size, 3), dtype=tf.int32)
    for _ in range(3):
        _ = movenet(warm)

    # Smoothing state (EMA) for head and shoulder center
    ema_head = None
    ema_sh = None
    alpha = 0.7  # higher = smoother

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Prepare tensor: resize with pad to 192x192
            img_tensor = tf.convert_to_tensor(frame_rgb)
            img_tensor = tf.expand_dims(img_tensor, axis=0)
            img_tensor = tf.cast(tf.image.resize_with_pad(img_tensor, target_size, target_size), dtype=tf.int32)

            # Run inference
            outputs = movenet(img_tensor)
            # shape [1,1,51] -> 17 keypoints * (y,x,score)
            kps = outputs['output_0'].numpy()[0, 0].reshape((17, 3))

            # For visualization, use the resized image so coords align
            vis_img = img_tensor[0].numpy().astype(np.uint8)
            vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            vis_img_bgr = draw_keypoints_on_image(vis_img_bgr, kps, confidence_threshold=0.3)

            current_distance = None
            nose = kps[0]
            l_sh = kps[5]
            r_sh = kps[6]


            if nose[2] > 0.3 and l_sh[2] > 0.3 and r_sh[2] > 0.3:
                head_xy = np.array([nose[1], nose[0]])
                shoulders_xy = (np.array([l_sh[1], l_sh[0]]) + np.array([r_sh[1], r_sh[0]])) / 2.0
                # Smooth
                if ema_head is None:
                    ema_head = head_xy
                    ema_sh = shoulders_xy
                else:
                    ema_head = alpha * ema_head + (1 - alpha) * head_xy
                    ema_sh = alpha * ema_sh + (1 - alpha) * shoulders_xy
                head_xy = ema_head
                shoulders_xy = ema_sh


                # Distance in normalized units -> pixels of target_size
                dist_norm = float(np.linalg.norm(head_xy - shoulders_xy))
                current_distance = dist_norm * float(target_size)


                # Tilt angle (degrees) between shoulder-center->head vector and vertical
                vec = head_xy - shoulders_xy
                angle_rad = np.arctan2(vec[0], -vec[1])  # x vs -y to compare with vertical
                angle_deg = float(np.degrees(angle_rad))
                angle_deg = abs(angle_deg)


                # Shoulder line tilt
                shoulder_tilt_deg = abs(float(np.degrees(np.arctan2(r_sh[1] - l_sh[1], r_sh[0] - l_sh[0]))))
                
                h, w = target_size, target_size
                p_head = (int(head_xy[0] * w), int(head_xy[1] * h))
                p_sh = (int(shoulders_xy[0] * w), int(shoulders_xy[1] * h))
                cv2.line(vis_img_bgr, p_head, p_sh, (0, 255, 255), 2)
                cv2.circle(vis_img_bgr, p_head, 4, (0, 255, 255), -1)
                cv2.circle(vis_img_bgr, p_sh, 4, (0, 255, 255), -1)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and current_distance is not None:
                baseline_distance = current_distance
                bad_posture_frames = 0
                good_posture_frames = 0
                posture_status = "Calibrated"
            elif key == ord('r'):
                baseline_distance = None
                bad_posture_frames = 0
                good_posture_frames = 0
                posture_status = "Reset"
            elif key == ord('q'):
                break

            # Posture evaluation
            if baseline_distance is not None and current_distance is not None:
                ratio = current_distance / baseline_distance


                excessive_tilt = False
                if 'angle_deg' in locals():
                    excessive_tilt = angle_deg > 12 or shoulder_tilt_deg > 10
                if ratio < 0.8 and excessive_tilt:
                    bad_posture_frames += 1
                    good_posture_frames = 0
                else:
                    good_posture_frames += 1
                    bad_posture_frames = 0
                if bad_posture_frames > 8:
                    posture_status = "Bad posture"
                elif good_posture_frames > 12:
                    posture_status = "Good posture"
                cv2.putText(vis_img_bgr, f"dist: {current_distance:.0f}px base: {baseline_distance:.0f}px ratio: {ratio:.2f}",
                            (8, target_size - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                if 'angle_deg' in locals():
                    cv2.putText(vis_img_bgr, f"head tilt: {angle_deg:.1f}\u00b0 shoulders: {shoulder_tilt_deg:.1f}\u00b0",
                                (8, target_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # FPS calculation
            frame_count += 1
            now = time.time()
            if now - last_time >= 0.5:
                displayed_fps = frame_count / (now - last_time)
                frame_count = 0
                last_time = now

            # Overlay UI
            cv2.putText(vis_img_bgr, f"FPS: {displayed_fps:.1f}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            if baseline_distance is None:
                cv2.putText(vis_img_bgr, "Press 'c' to calibrate", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
            else:
                color = (0, 0, 255) if posture_status == "Bad posture" else (0, 255, 0)
                cv2.putText(vis_img_bgr, f"Posture: {posture_status}", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            cv2.putText(vis_img_bgr, "q: quit  r: reset", (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow('MoveNet Multipose (press q to quit)', vis_img_bgr)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
