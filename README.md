# DocScan

System skanowania dokumentów i rozpoznawania tekstu (OCR) z archiwum w bazie danych SQLite.
Obsługiwane systemy operacyjne: Windows, Linux, macOS.

## Wymagania

- Python 3.10 lub nowszy
- Program Tesseract OCR wraz z pakietem językowym polskim:
  - Windows: instalator UB-Mannheim (podczas instalacji należy wybrać język Polish)
  - Arch Linux: `sudo pacman -S tesseract tesseract-data-pol tesseract-data-eng tk`
  - Debian/Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-pol tesseract-ocr-eng python3-tk`

## Uruchomienie

- Linux/macOS: `./run.sh`
- Windows: `run.bat`

Skrypty automatycznie tworzą środowisko wirtualne i instalują zależności z pliku `requirements.txt`.

## Funkcje

- Wykonanie zdjęcia kamerą lub wczytanie obrazu z pliku.
- Automatyczne wykrycie krawędzi dokumentu, kadrowanie i korekta perspektywy.
- Ręczna korekta narożników poprzez ich przeciągnięcie.
- Tryby obrazu: kolor, skala szarości, czarno-biały.
- Rozpoznawanie tekstu (Tesseract) z opcjonalnym filtrowaniem błędnie rozpoznanych znaków.
- Eksport skanu (PNG, JPG, PDF) oraz tekstu (TXT).
- Archiwum: zapis oryginału, skanu i tekstu w bazie danych wraz z tagami oraz wyszukiwanie według tagów.

## Lokalizacja bazy danych

- Windows: `%APPDATA%\DocScan\docscan.db`
- Linux: `~/.local/share/DocScan/docscan.db`
- macOS: `~/Library/Application Support/DocScan/docscan.db`
