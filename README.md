# PDF Summary & Explainer

Upload a PDF in a local browser page to see a short summary and an overview
assembled from the PDF's key sentences. Scanned pages are rendered and read with
the local OCR engine. The app does not require an API key.

## Start the app on Windows

Double-click `run.bat`. It starts the app and opens a local Streamlit page in
your browser. If the browser does not open automatically, copy the `Local URL`
shown in the terminal (normally `http://localhost:8501`) into your browser.

You can also start it from PowerShell in the project folder:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Then choose a PDF, select a summary length, and click **Summarize and explain
PDF**.

## Notes

- The summary and explanation are made from sentences extracted from the PDF;
  this version does not use a generative AI model.
- Scanned PDFs use RapidOCR. The first OCR run may take longer while OCR models
  are initialized.
- Scanned PDFs on Windows require the **Microsoft Visual C++ 2015-2022
  Redistributable (x64)** for OCR. Regular selectable-text PDFs do not need it.
- To install dependencies in the project's virtual environment, run
  `.\venv\Scripts\python.exe -m pip install -r requirements.txt`.
