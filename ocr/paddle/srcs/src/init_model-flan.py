from ocr_manager_faln import OCRManager

ocr_manager = OCRManager()

# for i in range(4341, 4342):
    # with open(f"/home/jupyter/advanced/ocr/sample_{102}.jpg", 'rb') as file:
with open('testing.jpg', 'rb') as file:
    img_bytes = file.read()

print(ocr_manager.ocr(img_bytes))   
