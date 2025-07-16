import cv2
import numpy as np

def mask_extracting(img, model):
  test = model.predict(source=img, save=True, imgsz=640)
  read_im = img.copy()
  try:
    if len(test[0].boxes.data) == 0:
      return None, read_im, None, None

    box = test[0].boxes
    conf = box.conf.data[0]

    mask = test[0].masks.data[0].cpu().numpy()
    mask = (mask >= 0.3).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
  
    read_im_resized = cv2.resize(read_im, (mask.shape[1], mask.shape[0]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blank_mask = np.zeros_like(mask, dtype=np.uint8)
  
    for cnt in contours:
        cv2.drawContours(blank_mask, [cnt], -1, color=1, thickness=-1)
  
    return blank_mask * 255, read_im_resized, contours
  except Exception as e:
    print(e)
    return None, read_im, None, None


def cropping_plate(img, mask, cont):
  largest_contour = max(cont, key=cv2.contourArea)
  x, y, w, h = cv2.boundingRect(largest_contour)

  cropped_image = img[y:y+h, x:x+w]
  cropped_mask = mask[y:y+h, x:x+w]
  alpha = np.where(cropped_mask > 0, 255, 0).astype(np.uint8)  
  plate_bgra = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2RGBA)
  plate_bgra[:, :, 3] = alpha

  
  return plate_bgra
