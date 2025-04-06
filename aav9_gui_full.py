import streamlit as st
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
from collections import Counter
from Bio import SeqIO, pairwise2
from Bio.Seq import Seq
from Bio.SubsMat import MatrixInfo

st.set_page_config(page_title="AAV9 Mutation Analyzer", layout="wide")
st.title("🧬 AAV9 Mutation Analyzer")

ref_seq = Seq("""
ATGGCTGCCGATGGTTATCTTCCAGATTGGCTCGAGGACAACCTTAGTGAAGGAATTCGCGAGTGGTGGGCTTTGAAACCTGGAGCCCCTCAACCCAAGGCAAATCAACAACATCAAGACAACGCTCGAGGTCTTGTGCTTCCGGGTTACAAATACCTTGGACCCGGCAACGGACTCGACAAGGGGGAGCCGGTCAACGCAGCAGACGCGGCGGCCCTCGAGCACGACAAGGCCTACGACCAGCAGCTCAAGGCCGGAGACAACCCGTACCTCAAGTACAACCACGCCGACGCCGAGTTCCAGGAGCGGCTCAAAGAAGATACGTCTTTTGGGGGCAACCTCGGGCGAGCAGTCTTCCAGGCCAAAAAGAGGCTTCTTGAACCTCTTGGTCTGGTTGAGGAAGCGGCTAAGACGGCTCCTGGAAAGAAGAGGCCTGTAGAGCAGTCTCCTCAGGAACCGGACTCCTCCGCGGGTATTGGCAAATCGGGTGCACAGCCCGCTAAAAAGAGACTCAATTTCGGTCAGACTGGCGACACAGAGTCAGTCCCAGACCCTCAACCAATCGGAGAACCTCCCGCAGCCCCCTCAGGTGTGGGATCTCTTACAATGGCTTCAGGTGGTGGCGCACCAGTGGCAGACAATAACGAAGGTGCCGATGGAGTGGGTAGTTCCTCGGGAAATTGGCATTGCGATTCCCAATGGCTGGGGGACAGAGTCATCACCACCAGCACCCGAACCTGGGCCCTGCCCACCTACAACAATCACCTCTACAAGCAAATCTCCAACAGCACATCTGGAGGATCTTCAAATGACAACGCCTACTTCGGCTACAGCACCCCCTGGGGGTATTTTGACTTCAACAGATTCCACTGCCACTTCTCACCACGTGACTGGCAGCGACTCATCAACAACAACTGGGGATTCCGGCCTAAGCGACTCAACTTCAAGCTCTTCAACATTCAGGTCAAAGAGGTTACGGACAACAATGGAGTCAAGACCATCGCCAATAACCTTACCAGCACGGTCCAGGTCTTCACGGACTCAGACTATCAGCTCCCGTACGTGCTCGGGTCGGCTCACGAGGGCTGCCTCCCGCCGTTCCCAGCGGACGTTTTCATGATTCCTCAGTACGGGTATCTGACGCTTAATGATGGAAGCCAGGCCGTGGGTCGTTCGTCCTTTTACTGCCTGGAATATTTCCCGTCGCAAATGCTAAGAACGGGTAACAACTTCCAGTTCAGCTACGAGTTTGAGAACGTACCTTTCCATAGCAGCTACGCTCACAGCCAAAGCCTGGACCGACTAATGAATCCACTCATCGACCAATACTTGTACTATCTCTCAAAGACTATTAACGGTTCTGGACAGAATCAACAAACGCTAAAATTCAGTGTGGCCGGACCCAGCAACATGGCTGTCCAGGGAAGAAACTACATACCTGGACCCAGCTACCGACAACAACGTGTCTCAACCACTGTGACTCAAAACAACAACAGCGAATTTGCTTGGCCTGGAGCTTCTTCTTGGGCTCTCAATGGACGTAATAGCTTGATGAATCCTGGACCTGCTATGGCCAGCCACAAAGAAGGAGAGGACCGTTTCTTTCCTTTGTCTGGATCTTTAATTTTTGGCAAACAAGGAACTGGAAGAGACAACGTGGATGCGGACAAAGTCATGATAACCAACGAAGAAGAAATTAAAACTACTAACCCGGTAGCAACGGAGTCCTATGGACAAGTGGCCACAAACCACCAGAGTGCCCAAGCACAGGCGCAGACCGGCTGGGTTCAAAACCAAGGAATACTTCCGGGTATGGTTTGGCAGGACAGAGATGTGTACCTGCAAGGACCCATTTGGGCCAAAATTCCTCACACGGACGGCAACTTTCACCCTTCTCCGCTGATGGGAGGGTTTGGAATGAAGCACCCGCCTCCTCAGATCCTCATCAAAAACACACCTGTACCTGCGGATCCTCCAACGGCCTTCAACAAGGACAAGCTGAACTCTTTCATCACCCAGTATTCTACTGGCCAAGTCAGCGTGGAGATCGAGTGGGAGCTGCAGAAGGAAAACAGCAAGCGCTGGAACCCGGAGATCCAGTACACTTCCAACTATTACAAGTCTAATAATGTTGAATTTGCTGTTAATACTGAAGGTGTATATAGTGAACCCCGCCCCATTGGCACCAGATACCTGACTCGTAATCTGTAA
""")

def annotate_mutations(ref, sample):
    mutations = []
    codon_pos = 0
    for i in range(0, min(len(ref), len(sample)) - 2, 3):
        ref_codon = ref[i:i+3]
        sample_codon = sample[i:i+3]
        if len(ref_codon) != 3 or len(sample_codon) != 3:
            continue
        if not all(base in "ATGCatgc" for base in ref_codon + sample_codon):
            continue
        try:
            ref_aa = str(Seq(ref_codon).translate())
            sample_aa = str(Seq(sample_codon).translate())
            if ref_aa != sample_aa:
                mutation = f"{ref_aa}{codon_pos+1}{sample_aa}"
                mutations.append(mutation)
        except:
            continue
        codon_pos += 1
    return mutations

def blosum_score(mutation):
    match = re.match(r"([A-Z])([0-9]+)([A-Z*])", mutation)
    if not match: return None
    from_aa, _, to_aa = match.groups()
    if to_aa == "*": return -10
    pair = (from_aa, to_aa)
    return MatrixInfo.blosum62.get(pair) or MatrixInfo.blosum62.get(pair[::-1]) if (pair[::-1] in MatrixInfo.blosum62) else None

uploaded_files = st.file_uploader("Upload your .ab1 files", type=["ab1"], accept_multiple_files=True)

if uploaded_files:
    mutation_matrix = {}
    all_mutations = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        for f in uploaded_files:
            path = os.path.join(tmpdirname, f.name)
            with open(path, "wb") as temp_file:
                temp_file.write(f.read())
            try:
                record = SeqIO.read(path, "abi")
                seq = record.seq
                if len(seq) < 50:
                    continue
                alignment = pairwise2.align.localms(ref_seq, seq, 2, -1, -5, -0.5, one_alignment_only=True)[0]
                aligned_ref = alignment.seqA.replace("-", "")
                aligned_seq = alignment.seqB.replace("-", "")
                muts = annotate_mutations(aligned_ref, aligned_seq)
                mutation_matrix[f.name] = muts
                all_mutations.extend(muts)
            except Exception as e:
                st.warning(f"{f.name} failed: {e}")

    if all_mutations:
        st.success("Mutations extracted successfully!")
        mutation_counts = Counter(all_mutations)
        df = pd.DataFrame(mutation_counts.items(), columns=["Mutation", "Frequency"]).sort_values("Frequency", ascending=False)

        df["BLOSUM62_Score"] = df["Mutation"].apply(blosum_score)
        df["Effect_Prediction"] = df["BLOSUM62_Score"].apply(
            lambda s: "Highly Disruptive" if s is not None and s <= -3
            else "Moderate" if s is not None and -3 < s < 1
            else "Likely Tolerated" if s is not None and s >= 1
            else "Unknown"
        )

        st.subheader("📊 Summary of All Mutations")
        st.dataframe(df)
        st.download_button("📥 Download Mutation Summary", df.to_csv(index=False).encode(), "mutation_summary.csv", "text/csv")

        st.subheader("📁 Per-Sample Mutation Reports")
        for sample, muts in mutation_matrix.items():
            rows = []
            for mut in muts:
                row = df[df["Mutation"] == mut]
                if not row.empty:
                    rows.append(row.iloc[0])
                else:
                    rows.append(pd.Series({"Mutation": mut, "Frequency": None, "BLOSUM62_Score": None, "Effect_Prediction": "Unknown"}))
            sample_df = pd.DataFrame(rows)
            st.markdown(f"**{sample}**")
            st.dataframe(sample_df)
            csv = sample_df.to_csv(index=False).encode()
            st.download_button(f"Download {sample}", csv, f"{sample}_mutation_report.csv", "text/csv")
    else:
        st.warning("No mutations found.")
