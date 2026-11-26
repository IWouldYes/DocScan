# DocScan

Skaner dokumentów: zdjęcie/plik -> wykrycie krawędzi -> prostowanie -> OCR (Tesseract).
Działa na Linux, macOS i Windows.

## Instalacja

1. Zainstaluj Tesseract z językiem polskim:
   - Arch: `sudo pacman -S tesseract tesseract-data-pol tesseract-data-eng tk`
   - Debian/Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-pol tesseract-ocr-eng python3-tk`
   - Windows: instalator UB-Mannheim (zaznacz Polish)
2. Uruchom:
   - Linux/macOS: `./run.sh`
   - Windows: `run.bat`

Skrypty same tworzą środowisko wirtualne i instalują zależności.

## Użycie
- 📷 Zrób zdjęcie / 📁 Wczytaj plik
- Rogi kartki można przeciągać myszą w zakładce „Oryginał”
- Tryby: Kolor / Szary / Czarno-biały, obrót o 90°
- Eksport: PNG/JPG/PDF oraz tekst .txt
