# Progetto d'esame Visual Information Processing Management

Questo progetto risolve tutti i task assegnati come esame.

## Documentazione

Una descrizione dettagliata del progetto è visualizzabile all'interno della documentazione nella directory ```/presentazione/```.


## Autori

- **Federica Ratti** — 886158
- **Nicolò Molteni** — 933190

Corso di Laurea Magistrale in Informatica  
Università degli Studi di Milano-Bicocca  
A.A. 2025-2026

---

## Setup del progetto

### 1. Clona la repository

Dopo aver creato una directory spostarsi al suo interno per clonare la repository.

```bash
cd existing_repo
git clone https://github.com/niki-molte/sport_classification
```

### 2. Setup del virtual environmnet

Aprire la directory del progetto nel terminale ed eseguire:

```bash
    python3 -m venv venv
    source venv/bin/activate       # Linux/macOS
    venv\Scripts\activate.bat      # Windows
```

### 3. Installa le dependencies

Dopo aver attivato il virtual environment è possibile installare le dependencies.

```bash
    pip install --upgrade pip
    pip install -r requirements.txt
```  

# Run

Aprire un terminale nella stessa schermata della directory in cui è stata clonata la repository ed eseguire

```bash
    streamlit run gui.py
```  

In questo verrà eseguita l'interfaccia grafica che permette il caricamento dell'immagine e del modello per risolvere i task. Per maggiori informazioni consultare le presentazione.

---
