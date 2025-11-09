import gzip

def load_utr_coords(gtf_path):
    """
    Parse an Ensembl GTF file to extract 5' and 3' UTR coordinates for genes.
    Returns a dict mapping gene_id -> {
        'chr': chromosome,
        'strand': strand,
        '5utr': [(start, end), ...],  # list of 5' UTR segments (genomic intervals)
        '3utr': [(start, end), ...],  # list of 3' UTR segments
        'min_start': int,            # min start of any UTR segment (for quick overlap checks)
        'max_end': int              # max end of any UTR segment
    } for a representative transcript of each gene (chosen as the one with longest total UTR length).
    """
    utr_coords = {}
    gene_full_span = {}  # store full gene span (start, end) from gene records for reference
    # We will gather UTR segments per transcript to choose best transcript per gene
    transcripts = {}  # transcript_id -> dict(gene, chr, strand, utr5_segments, utr3_segments)
    # Also gather gene full spans if gene lines present
    opener = gzip.open if str(gtf_path).endswith((".gz", ".bgz")) else open
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue
            chrom, source, feature, start, end, score, strand, frame, attrs = fields
            if feature == "gene":
                # Parse attributes to get gene_id
                gene_id = None
                for attr in attrs.split(";"):
                    attr = attr.strip()
                    if not attr:
                        continue
                    key, val = attr.split(" ", 1)
                    if key == "gene_id":
                        gene_id = val.strip().strip('"')
                        break
                if gene_id:
                    gene_full_span[gene_id] = (chrom, int(start), int(end), strand)
            elif feature in ("five_prime_utr", "three_prime_utr", "five_prime_UTR", "three_prime_UTR"):
                # Parse gene_id and transcript_id from attributes
                gene_id = None
                transcript_id = None
                for attr in attrs.split(";"):
                    attr = attr.strip()
                    if not attr:
                        continue
                    key, val = attr.split(" ", 1)
                    val = val.strip().strip('"')
                    if key == "gene_id":
                        gene_id = val
                    elif key == "transcript_id":
                        transcript_id = val
                if gene_id is None or transcript_id is None:
                    continue
                # Initialize transcript record if not present
                if transcript_id not in transcripts:
                    transcripts[transcript_id] = {
                        "gene": gene_id,
                        "chr": chrom,
                        "strand": strand,
                        "utr5_segments": [],
                        "utr3_segments": []
                    }
                # Add this UTR segment
                seg_start, seg_end = int(start), int(end)
                if feature.lower().startswith("five"):
                    transcripts[transcript_id]["utr5_segments"].append((seg_start, seg_end))
                elif feature.lower().startswith("three"):
                    transcripts[transcript_id]["utr3_segments"].append((seg_start, seg_end))
    # Choose one transcript per gene (with longest total UTR length)
    best_transcript_for_gene = {}
    for tid, info in transcripts.items():
        gene = info["gene"]
        # Calculate total UTR length
        total_len = 0
        for (s, e) in info["utr5_segments"]:
            total_len += (e - s + 1)
        for (s, e) in info["utr3_segments"]:
            total_len += (e - s + 1)
        if gene not in best_transcript_for_gene or total_len > best_transcript_for_gene[gene]["length"]:
            best_transcript_for_gene[gene] = {"transcript_id": tid, "length": total_len}
    # Build utr_coords for each gene using the best transcript's UTR segments
    for gene, best_info in best_transcript_for_gene.items():
        tid = best_info["transcript_id"]
        info = transcripts[tid]
        chrom = info["chr"]
        strand = info["strand"]
        seg5 = sorted(info["utr5_segments"])  # sort by start
        seg3 = sorted(info["utr3_segments"])
        if not seg5 and not seg3:
            # If no UTR segments at all (shouldn't happen for typical mRNA), skip
            continue
        # Compute min and max coordinates of all UTR segments
        all_starts = [s for (s, e) in seg5 + seg3]
        all_ends = [e for (s, e) in seg5 + seg3]
        min_start = min(all_starts) if all_starts else None
        max_end = max(all_ends) if all_ends else None
        utr_coords[gene] = {
            "chr": chrom,
            "strand": strand,
            "5utr": seg5,
            "3utr": seg3,
            "min_start": min_start,
            "max_end": max_end
        }
        # If we have gene full span (from gene annotation) use that for safety in distance calc (below)
        if gene in gene_full_span:
            g_chr, g_start, g_end, g_strand = gene_full_span[gene]
            # We expect g_chr equals chrom
            utr_coords[gene]["gene_start"] = int(g_start)
            utr_coords[gene]["gene_end"] = int(g_end)
        else:
            # If no gene record, use UTR min/max as proxy for gene span
            utr_coords[gene]["gene_start"] = min_start
            utr_coords[gene]["gene_end"] = max_end
    return utr_coords

def load_rbp_features(eclip_dir, utr_coords):
    """
    Load RBP (RNA-binding protein) peak features from ENCODE eCLIP data.
    eclip_dir should contain .bed.gz files, one per cell line (e.g., 'HepG2.bed.gz', 'K562.bed.gz').
    Each file has peak intervals (chr, start, end, ...) with a signal value.
    utr_coords is the dict from load_utr_coords, providing UTR segment coords for each gene.
    Returns rbp_features: dict[cell_line][gene_id] = {
        'count_5utr': count of peaks overlapping 5' UTR,
        'avg_signal_5utr': average peak signal in 5' UTR (0 if no peaks),
        'count_3utr': count of peaks in 3' UTR,
        'avg_signal_3utr': average signal in 3' UTR
    }
    """
    import os
    # Pre-index genes by chromosome for faster overlap lookup
    genes_by_chr = {}
    for gene, info in utr_coords.items():
        chrom = info["chr"]
        genes_by_chr.setdefault(chrom, []).append((gene, info))
    # Sort gene lists by starting coordinate for each chromosome
    for chrom in genes_by_chr:
        genes_by_chr[chrom].sort(key=lambda x: x[1]["min_start"] if x[1]["min_start"] is not None else 0)
    rbp_features = {}
    for fname in os.listdir(eclip_dir):
        if not fname.endswith(".bed.gz"):
            continue
        cell_line = fname.replace(".bed.gz", "")
        rbp_features[cell_line] = {}
        file_path = os.path.join(eclip_dir, fname)
        with gzip.open(file_path, "rt") as f:
            # Read peaks and assign to genes
            for line in f:
                if line.startswith("#") or line.strip() == "":
                    continue
                cols = line.strip().split()
                if len(cols) < 3:
                    continue
                chrom = cols[0]
                peak_start = int(cols[1])
                peak_end = int(cols[2])
                # Determine signal value from columns (supports standard BED or narrowPeak):
                if len(cols) >= 7:
                    # narrowPeak format (assume 7th column is signal)
                    try:
                        signal = float(cols[6])
                    except:
                        signal = float(cols[4]) if len(cols) > 4 else 1.0
                else:
                    # BED format (use score if available)
                    signal = float(cols[4]) if len(cols) > 4 else 1.0
                if chrom not in genes_by_chr:
                    continue
                # Iterate relevant genes on this chromosome that could overlap this peak
                for gene, info in genes_by_chr[chrom]:
                    # Quick elimination: if gene's UTR region is far from this peak
                    if info["min_start"] is not None and peak_end < info["min_start"]:
                        # Since genes_by_chr list is sorted by min_start, if this gene's min_start is beyond peak_end,
                        # then all subsequent genes will also start after this peak. We can break out early.
                        break
                    if info["max_end"] is not None and peak_start > info["max_end"]:
                        # Peak is entirely to the left of this gene's UTR range, continue to next gene.
                        continue
                    # Now check overlap with any UTR segments
                    count5 = 0
                    count3 = 0
                    sum5 = 0.0
                    sum3 = 0.0
                    # Check 5' UTR segments
                    for (s, e) in info["5utr"]:
                        if peak_end < s or peak_start > e:
                            continue  # no overlap with this segment
                        # Overlap exists
                        count5 = 1  # mark as hit (we won't double count a single peak multiple times in the same UTR)
                        sum5 = signal
                        break  # one peak counts once per UTR type
                    # Check 3' UTR segments
                    for (s, e) in info["3utr"]:
                        if peak_end < s or peak_start > e:
                            continue
                        count3 = 1
                        sum3 = signal
                        break
                    if count5 == 0 and count3 == 0:
                        continue  # peak did not hit any UTR segment of this gene
                    # If we found a hit, accumulate into gene's record in rbp_features
                    rec = rbp_features[cell_line].setdefault(gene, {
                        "count_5utr": 0, "sum_signal_5utr": 0.0,
                        "count_3utr": 0, "sum_signal_3utr": 0.0
                    })
                    rec["count_5utr"] += count5
                    rec["sum_signal_5utr"] += sum5
                    rec["count_3utr"] += count3
                    rec["sum_signal_3utr"] += sum3
        # After reading all peaks for this cell_line, compute average signals
        for gene, rec in rbp_features[cell_line].items():
            c5 = rec["count_5utr"]
            c3 = rec["count_3utr"]
            rec["avg_signal_5utr"] = rec["sum_signal_5utr"] / c5 if c5 > 0 else 0.0
            rec["avg_signal_3utr"] = rec["sum_signal_3utr"] / c3 if c3 > 0 else 0.0
            # Remove the temporary sum fields
            del rec["sum_signal_5utr"], rec["sum_signal_3utr"]
    return rbp_features

def load_trna_features(trna_bed_path, utr_coords):
    """
    Compute tRNA proximity features for each gene.
    trna_bed_path: BED file containing genomic intervals of all tRNA loci.
    utr_coords: dict from load_utr_coords with gene coordinates.
    Returns trna_features: dict[gene_id] = {'tRNA_overlap': 0/1, 'tRNA_distance': distance_to_nearest_tRNA}
    If a tRNA overlaps the gene region (between gene_start and gene_end inclusive), 'tRNA_overlap' will be 1 and distance = 0.
    Otherwise 'tRNA_overlap' = 0 and 'tRNA_distance' is the distance in base pairs from the nearest tRNA locus (upstream or downstream).
    """
    trna_coords_by_chr = {}
    # Read tRNA coordinates
    with open(trna_bed_path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            cols = line.strip().split()
            if len(cols) < 3:
                continue
            chrom = cols[0]
            start = int(cols[1])
            end = int(cols[2])
            trna_coords_by_chr.setdefault(chrom, []).append((start, end))
    # Sort tRNA intervals by start per chromosome
    for chrom in trna_coords_by_chr:
        trna_coords_by_chr[chrom].sort(key=lambda x: x[0])
    trna_features = {}
    for gene, info in utr_coords.items():
        # Use gene full span for distance calculation
        if "gene_start" in info and "gene_end" in info:
            gene_start = info["gene_start"]
            gene_end = info["gene_end"]
        else:
            # Fallback: use UTR min/max if full gene span not available
            gene_start = info.get("min_start", None)
            gene_end = info.get("max_end", None)
        if gene_start is None or gene_end is None:
            continue  # skip if we don't have coordinates
        chrom = info["chr"]
        overlap_flag = 0
        nearest_dist = None
        if chrom in trna_coords_by_chr:
            for (t_start, t_end) in trna_coords_by_chr[chrom]:
                # If tRNA interval overlaps gene interval
                if not (t_end < gene_start or t_start > gene_end):
                    overlap_flag = 1
                    nearest_dist = 0
                    break
                # Compute distance if not overlapping
                if t_end < gene_start:
                    dist = gene_start - t_end  # tRNA is upstream
                else:
                    dist = t_start - gene_end  # tRNA is downstream
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
        # If no tRNA on same chromosome or none found, set a large distance
        if nearest_dist is None:
            nearest_dist = 1e6  # use a sentinel large distance (e.g., 1e6 bp) to indicate no nearby tRNA
        trna_features[gene] = {
            "tRNA_overlap": 1 if overlap_flag else 0,
            "tRNA_distance": float(nearest_dist)
        }
    return trna_features

def load_tissue_embeddings(path):
    """
    Load tissue embedding vectors from an external file.
    The file is expected to contain a dict or other mapping from tissue identifier to embedding vector.
    Returns a dict: {tissue_id: embedding_vector} (embedding_vector can be list or numpy array).
    """
    import pickle
    # Here we assume the embeddings are stored in a pickle file for simplicity.
    with open(path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings
