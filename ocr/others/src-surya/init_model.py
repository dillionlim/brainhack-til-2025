from ocr_manager import OCRManager

ocr_manager = OCRManager()

with open('testing.jpg', 'rb') as file:
    img_bytes = file.read()

print(ocr_manager.ocr([img_bytes]))    
