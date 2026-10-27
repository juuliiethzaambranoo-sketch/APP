# Laboratorio SpVAR – Colombia (Streamlit)

App de econometría espacial: IPM ↔ conectividad, Moran, LISA, modelo SAR, mapas y vista 3D.

## Archivos
- `app.py` – la aplicación
- `requirements.txt` – dependencias
- `README.md` – este archivo

## Correr local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Subir a Streamlit Community Cloud
1. Crea un repositorio en GitHub y sube `app.py`, `requirements.txt` y `README.md` a la raíz.
2. Entra a https://share.streamlit.io e inicia sesión con GitHub.
3. **Create app** → elige el repositorio, la rama (`main`) y el archivo principal `app.py`.
4. En **Advanced settings** selecciona Python 3.11 o 3.12 y pulsa **Deploy**.

La primera carga tarda 2–5 minutos mientras se instalan las librerías geoespaciales.

## Notas
- Los datos del GeoJSON se descargan de GitHub al iniciar; se guardan en caché con `st.cache_data`.
- Si el modelo SAR no converge, la app usa OLS automáticamente.
