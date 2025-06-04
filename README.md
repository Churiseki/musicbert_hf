# MusicBERT Embedding Extractor

This project is based on [musicbert_hf](https://github.com/malcolmsailor/musicbert_hf) and provides a simple pipeline to extract embeddings from MIDI files using MusicBERT.

## 🔗 Original Repository

> https://github.com/malcolmsailor/musicbert_hf

---

## 🚀 Usage

### Step 1: Prepare MIDI Input Files

Place your `.mid` files inside the `midis/` directory.

project-root/ \
├── midis/ \
│ ├── your_file1.mid \
│ └── your_file2.mid 


### Step 2: Install Dependencies

Make sure you are using Python 3.11, then install all required packages:

```bash
pip install -r requirements.txt
```
Step 3-0: Download the checkpoint:
https://drive.google.com/file/d/12B_Mhl3OS2z7DREEmXeG9nX1Nq-Tl8P4/view?usp=share_link

and put it in the same file as multi_ex.py


Step 3: Extract Embeddings
Run the following script to process the MIDI files and extract embeddings:
```bash
python multi_ex.py
```
This will generate a file called all_embeddings.pt containing the extracted embeddings.

Step 4: Read Embeddings
To read and manipulate the embeddings vectors, refer to:
```
read_embeddings_vec.py
```
This script shows how to load and work with the all_embeddings.pt data.