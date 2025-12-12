# dicom_processor.py
import pydicom
import numpy as np
from PIL import Image


class DicomProcessor:

    @staticmethod
    def read_dicom_file(file_path):
        """
        Citește un fișier DICOM și îl convertește în imagine pentru AI
        """
        try:
            # Citește fișierul DICOM
            ds = pydicom.dcmread(file_path)

            # Extrage datele pixel
            pixel_array = ds.pixel_array

            # Normalizare
            pixel_array = pixel_array.astype(float)
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
            pixel_array = (pixel_array * 255).astype(np.uint8)

            return pixel_array

        except Exception as e:
            print(f"Eroare la citirea DICOM: {e}")
            return None

    @staticmethod
    def preprocess_for_model(image_array, target_size=(224, 224)):
        """
        Preprocesează imaginea pentru modelul de AI
        """
        # Convertește la PIL Image
        img = Image.fromarray(image_array)

        # Resize
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Convertește înapoi la numpy
        img_array = np.array(img)

        # Normalizare pentru model
        img_array = img_array / 255.0

        # Adaugă dimensiuni dacă e necesar
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.repeat(img_array, 3, axis=-1)  # Convertește la RGB

        img_array = np.expand_dims(img_array, axis=0)  # Batch dimension

        return img_array