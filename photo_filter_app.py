import cv2
import time
import os
import numpy as np
from datetime import datetime

class PhotoFilterApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam. Please check your camera connection.")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.current_filter = "None"
        self.show_help = False
        self.blur_kernel_size = 5
        self.gaussian_sigma = 1.0
        self.canny_threshold1 = 100
        self.canny_threshold2 = 200
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
        self.face_detection_interval = 3
        self.frame_count = 0
        self.last_faces = []
        self.last_landmarks = []
        self.face_detection_scale = 0.5
        
        self.output_dir = "filtered_images"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.dnn_net = self._load_dnn_face_detector()
        self.use_dnn = self.dnn_net is not None
    
    def get_fps(self):
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_time = current_time
        return self.current_fps
    
    def detect_faces(self, frame):
        small_frame = cv2.resize(frame, None, fx=self.face_detection_scale, fy=self.face_detection_scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20)
        )
        faces = [[int(x/self.face_detection_scale), int(y/self.face_detection_scale), 
                  int(w/self.face_detection_scale), int(h/self.face_detection_scale)] 
                 for x, y, w, h in faces]
        return faces
    
    def _load_dnn_face_detector(self):
        try:
            model_file = cv2.data.haarcascades.replace('haarcascades', '') + '../dnn/opencv_face_detector_uint8.pb'
            config_file = cv2.data.haarcascades.replace('haarcascades', '') + '../dnn/opencv_face_detector.pbtxt'
            if os.path.exists(model_file) and os.path.exists(config_file):
                return cv2.dnn.readNetFromTensorflow(model_file, config_file)
        except:
            pass
        return None
    
    def detect_faces_dnn(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                face_w = x2 - x1
                face_h = y2 - y1
                if face_w > 20 and face_h > 20:
                    faces.append([x1, y1, face_w, face_h])
        return faces
    
    def estimate_facial_landmarks(self, frame, face_rect):
        x, y, w, h = face_rect
        face_region = frame[y:y+h, x:x+w]
        
        landmarks = {}
        landmarks['left_eye'] = (int(x + w * 0.30), int(y + h * 0.35))
        landmarks['right_eye'] = (int(x + w * 0.70), int(y + h * 0.35))
        landmarks['mouth'] = (int(x + w * 0.50), int(y + h * 0.75))
        landmarks['face_center'] = (int(x + w * 0.50), int(y + h * 0.50))
        
        return landmarks
    
    def get_face_mask(self, frame, face_rect):
        x, y, w, h = face_rect
        mask = np.zeros((h, w), dtype=np.uint8)
        
        center_x = w // 2
        center_y = h // 2
        axes_a = int(w * 0.45)
        axes_b = int(h * 0.55)
        
        cv2.ellipse(mask, (center_x, center_y), (axes_a, axes_b), 0, 0, 360, 255, -1)
        
        return mask
    
    def apply_box_blur(self, frame):
        kernel_size = self.blur_kernel_size
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.blur(frame, (kernel_size, kernel_size))
    
    def apply_gaussian_blur(self, frame):
        kernel_size = int(self.gaussian_sigma * 10) * 2 + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            kernel_size = 3
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), self.gaussian_sigma)
    
    def apply_sharpening(self, frame):
        blurred = cv2.GaussianBlur(frame, (0, 0), 2.0)
        return cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
    
    def apply_sobel(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobelx, sobely)
        magnitude = cv2.convertScaleAbs(magnitude)
        return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)
    
    def apply_canny(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    def apply_sepia_tone(self, image):
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia = cv2.transform(image, sepia_filter)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        return sepia
    
    def add_grainy_texture(self, image, intensity=30):
        noise = np.random.normal(0, intensity, image.shape).astype(np.int16)
        textured = image.astype(np.int16) + noise
        textured = np.clip(textured, 0, 255).astype(np.uint8)
        return textured
    
    def create_peanut_shaped_mask(self, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        center_x, center_y = w // 2, h // 2
        
        lobe_width = int(w * 0.35)
        lobe_height = int(h * 0.55)
        
        left_lobe_center = (int(center_x * 0.6), center_y)
        right_lobe_center = (int(center_x * 1.4), center_y)
        
        cv2.ellipse(mask, left_lobe_center, (lobe_width, lobe_height), 0, 0, 360, 255, -1)
        cv2.ellipse(mask, right_lobe_center, (lobe_width, lobe_height), 0, 0, 360, 255, -1)
        
        feather_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, feather_kernel)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        
        return mask
    
    def load_peanut_texture_image(self):
        texture_paths = [
            "peanut_texture.jpg",
            os.path.join(os.path.dirname(__file__), "peanut_texture.jpg"),
            os.path.join(self.output_dir, "..", "peanut_texture.jpg"),
            "d:/CLASS projects/peanut_texture.jpg"
        ]
        
        for texture_path in texture_paths:
            if os.path.exists(texture_path):
                texture = cv2.imread(texture_path)
                if texture is not None:
                    print(f"Loaded peanut texture from: {texture_path}")
                    return texture
        
        print("Peanut texture image not found. Using procedural texture.")
        return None
    
    def generate_peanut_texture(self, h, w):
        peanut_texture = self.load_peanut_texture_image()
        
        if peanut_texture is not None:
            resized_texture = cv2.resize(peanut_texture, (w, h))
            return resized_texture
        
        base_color_light = np.uint8([180, 170, 150])
        base_color_dark = np.uint8([140, 130, 110])
        
        peanut_layer = np.zeros((h, w, 3), dtype=np.uint8)
        
        for y in range(h):
            for x in range(w):
                blend = (x / w) * 0.3
                color = (base_color_light.astype(float) * (1 - blend) + 
                        base_color_dark.astype(float) * blend).astype(np.uint8)
                peanut_layer[y, x] = color
        
        noise = np.random.normal(0, 20, (h, w, 3)).astype(np.int16)
        peanut_layer = peanut_layer.astype(np.int16) + noise
        peanut_layer = np.clip(peanut_layer, 0, 255).astype(np.uint8)
        
        return peanut_layer
    
    def apply_peanut_ar_filter(self, frame):
        self.frame_count += 1
        
        if self.frame_count % self.face_detection_interval == 0:
            if self.use_dnn:
                self.last_faces = self.detect_faces_dnn(frame)
            else:
                self.last_faces = self.detect_faces(frame)
            
            self.last_landmarks = []
            for face_rect in self.last_faces:
                landmarks = self.estimate_facial_landmarks(frame, face_rect)
                self.last_landmarks.append(landmarks)
        
        if len(self.last_faces) == 0:
            return frame
        
        result = frame.copy()
        h, w = frame.shape[:2]
        
        for idx, face_rect in enumerate(self.last_faces):
            face_x, face_y, face_w, face_h = face_rect
            
            if face_x < 0 or face_y < 0 or face_x + face_w > w or face_y + face_h > h:
                continue
            
            landmarks = self.last_landmarks[idx] if idx < len(self.last_landmarks) else self.estimate_facial_landmarks(frame, face_rect)
            
            peanut_layer = self.generate_peanut_texture(face_h, face_w)
            
            peanut_mask = self.create_peanut_shaped_mask(face_h, face_w)
            
            peanut_mask_normalized = peanut_mask.astype(np.float32) / 255.0
            
            face_region = result[face_y:face_y+face_h, face_x:face_x+face_w]
            
            left_eye_x, left_eye_y = landmarks['left_eye']
            right_eye_x, right_eye_y = landmarks['right_eye']
            mouth_x, mouth_y = landmarks['mouth']
            
            eye_radius = int(face_w * 0.08)
            mouth_radius = int(face_w * 0.10)
            
            left_eye_local = (left_eye_x - face_x, left_eye_y - face_y)
            right_eye_local = (right_eye_x - face_x, right_eye_y - face_y)
            mouth_local = (mouth_x - face_x, mouth_y - face_y)
            
            eye_mask = np.zeros((face_h, face_w), dtype=np.uint8)
            cv2.circle(eye_mask, left_eye_local, eye_radius, 255, -1)
            cv2.circle(eye_mask, right_eye_local, eye_radius, 255, -1)
            cv2.circle(eye_mask, mouth_local, mouth_radius, 255, -1)
            
            eye_mask_normalized = eye_mask.astype(np.float32) / 255.0
            
            eyes_and_mouth = cv2.bitwise_and(face_region, face_region, mask=eye_mask)
            
            peanut_with_eyes = peanut_layer.copy()
            for c in range(3):
                peanut_with_eyes[:,:,c] = (peanut_layer[:,:,c].astype(np.float32) * (1 - eye_mask_normalized) + 
                                           face_region[:,:,c].astype(np.float32) * eye_mask_normalized).astype(np.uint8)
            
            blended = face_region.astype(np.float32).copy()
            for c in range(3):
                blended[:,:,c] = (face_region[:,:,c].astype(np.float32) * (1 - peanut_mask_normalized) + 
                                 peanut_with_eyes[:,:,c].astype(np.float32) * peanut_mask_normalized)
            
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            
            result[face_y:face_y+face_h, face_x:face_x+face_w] = blended
            
            cv2.rectangle(result, (face_x, face_y), (face_x+face_w, face_y+face_h), (0, 255, 0), 2)
        
        return result
    
    def apply_simple_peanut_filter(self, frame):
        self.frame_count += 1
        
        if self.frame_count % self.face_detection_interval == 0:
            self.last_faces = self.detect_faces(frame)
        
        if len(self.last_faces) == 0:
            return frame
        
        result = frame.copy()
        h, w = frame.shape[:2]
        
        for face_data in self.last_faces:
            face_x, face_y, face_w, face_h = face_data
            
            if face_x < 0 or face_y < 0 or face_x + face_w > w or face_y + face_h > h:
                continue
            
            peanut_mask = self.get_face_mask(frame, (face_x, face_y, face_w, face_h))
            
            peanut_color = np.uint8([100, 140, 180])
            peanut_overlay = np.full((face_h, face_w, 3), peanut_color, dtype=np.uint8)
            
            peanut_mask_3ch = cv2.cvtColor(peanut_mask, cv2.COLOR_GRAY2BGR)
            peanut_mask_3ch = peanut_mask_3ch.astype(float) / 255.0
            
            face_region = result[face_y:face_y+face_h, face_x:face_x+face_w]
            
            blended = (face_region.astype(float) * (1 - peanut_mask_3ch) + 
                      peanut_overlay.astype(float) * peanut_mask_3ch)
            blended = np.uint8(np.clip(blended, 0, 255))
            
            result[face_y:face_y+face_h, face_x:face_x+face_w] = blended
            
            cv2.rectangle(result, (face_x, face_y), (face_x+face_w, face_y+face_h), (0, 255, 0), 2)
        
        return result
    
    def draw_ui(self, frame):
        fps = self.get_fps()
        
        cv2.putText(frame, f"Filter: {self.current_filter}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.show_help:
            help_text = [
                "KEYBOARD SHORTCUTS:",
                "B - Box Blur (5x5/11x11)",
                "G - Gaussian Blur",
                "S - Sharpening",
                "E - Sobel Edge Detection",
                "C - Canny Edge Detection",
                "P - Peanut Filter (Simple)",
                "A - Peanut AR Filter (Advanced)",
                "N - No Filter",
                "H - Toggle Help",
                "SPACE - Save Image",
                "Q - Quit"
            ]
            y_offset = 100
            for text in help_text:
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
        
        return frame
    
    def save_image(self, frame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"filtered_{self.current_filter}_{timestamp}.png")
        cv2.imwrite(filename, frame)
        print(f"Image saved: {filename}")
    
    def run(self):
        print("Photo Filter Application Started")
        print("Press 'H' for help, 'Q' to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to read frame from webcam")
                    break
                
                frame = cv2.flip(frame, 1)
                
                if self.current_filter == "Box Blur":
                    frame = self.apply_box_blur(frame)
                elif self.current_filter == "Gaussian Blur":
                    frame = self.apply_gaussian_blur(frame)
                elif self.current_filter == "Sharpening":
                    frame = self.apply_sharpening(frame)
                elif self.current_filter == "Sobel":
                    frame = self.apply_sobel(frame)
                elif self.current_filter == "Canny":
                    frame = self.apply_canny(frame)
                elif self.current_filter == "Peanut":
                    frame = self.apply_simple_peanut_filter(frame)
                elif self.current_filter == "Peanut AR":
                    frame = self.apply_peanut_ar_filter(frame)
                
                frame = self.draw_ui(frame)
                
                cv2.imshow("Photo Filter Application", frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('b'):
                    self.blur_kernel_size = 11 if self.blur_kernel_size == 5 else 5
                    self.current_filter = f"Box Blur ({self.blur_kernel_size}x{self.blur_kernel_size})"
                elif key == ord('g'):
                    self.current_filter = "Gaussian Blur"
                elif key == ord('s'):
                    self.current_filter = "Sharpening"
                elif key == ord('e'):
                    self.current_filter = "Sobel"
                elif key == ord('c'):
                    self.current_filter = "Canny"
                elif key == ord('p'):
                    self.current_filter = "Peanut"
                elif key == ord('a'):
                    self.current_filter = "Peanut AR"
                elif key == ord('n'):
                    self.current_filter = "None"
                elif key == ord('h'):
                    self.show_help = not self.show_help
                elif key == ord(' '):
                    self.save_image(frame)
                elif key == ord('+') or key == ord('='):
                    if self.current_filter == "Gaussian Blur":
                        self.gaussian_sigma = min(5.0, self.gaussian_sigma + 0.1)
                    elif self.current_filter == "Canny":
                        self.canny_threshold1 = min(200, self.canny_threshold1 + 10)
                        self.canny_threshold2 = min(255, self.canny_threshold2 + 10)
                elif key == ord('-') or key == ord('_'):
                    if self.current_filter == "Gaussian Blur":
                        self.gaussian_sigma = max(0.1, self.gaussian_sigma - 0.1)
                    elif self.current_filter == "Canny":
                        self.canny_threshold1 = max(10, self.canny_threshold1 - 10)
                        self.canny_threshold2 = max(20, self.canny_threshold2 - 10)
        
        except KeyboardInterrupt:
            print("Application interrupted by user")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Application closed")

if __name__ == "__main__":
    try:
        app = PhotoFilterApp()
        app.run()
    except RuntimeError as e:
        print(f"Error: {e}")
