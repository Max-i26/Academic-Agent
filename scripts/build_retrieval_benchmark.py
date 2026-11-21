"""Build the 30-question retrieval benchmark with verified chunk IDs and relevance grades."""

import json
from pathlib import Path


def main():
    with open("data/evaluation/chunks_inventory.json", "r") as f:
        inventory = json.load(f)

    def search_chunks(keyword, file_filter=None):
        results = []
        for c in inventory:
            if file_filter and file_filter not in c["source_file"]:
                continue
            text = c["text"].lower() + " " + (c["section_title"] or "").lower()
            if keyword.lower() in text:
                results.append(c)
        return results

    # Helper to print chunk options for a query
    print("--- Searching Chunks for Benchmark Questions ---")

    # Let's inspect relevant chunks for key topics:
    mae_chunks = search_chunks("mae", "Lecture4")
    ses_chunks = search_chunks("ses equation", "Lecture4")
    decomp_chunks = search_chunks("multiplicative", "Lecture5")
    cma_chunks = search_chunks("centered moving average", "Lecture5")
    acf_bounds_chunks = search_chunks("significance bounds", "Lecture7")
    stationarity_chunks = search_chunks("why do we need stationarity", "Lecture6")
    diff_chunks = search_chunks("first difference", "Lecture6")
    whitenoise_chunks = search_chunks("white noise", "Lecture7")
    seasonal_calc_chunks = search_chunks("seasonal index", "Lecture5")
    cma_workspace_chunks = search_chunks("cma workspace", "Lecture5")
    ses_calc_chunks = search_chunks("worked example", "Lecture4")
    acf_lag_chunks = search_chunks("lag", "Lecture7")

    print(f"MAE chunks: {len(mae_chunks)}")
    print(f"SES chunks: {len(ses_chunks)}")
    print(f"Decomposition chunks: {len(decomp_chunks)}")
    print(f"CMA chunks: {len(cma_chunks)}")
    print(f"ACF bounds chunks: {len(acf_bounds_chunks)}")
    print(f"Stationarity chunks: {len(stationarity_chunks)}")
    print(f"Difference chunks: {len(diff_chunks)}")
    print(f"White noise chunks: {len(whitenoise_chunks)}")
    print(f"CMA workspace chunks: {len(cma_workspace_chunks)}")

    # Let's write out a script that precisely maps 30 queries with primary and secondary relevance
    # We will build the benchmark dictionary
    queries = [
        # =========================================================================
        # 1. Formulas and Mathematical Notation (5 questions)
        # =========================================================================
        {
            "query_id": "q01",
            "query": "What is the formula for Mean Absolute Error (MAE)?",
            "category": "formulas_and_notation",
            "difficulty": "easy",
            "relevant_records": {
                "7bcdd909a9a4d4ac": 3,  # Primary MAE slide
            },
            "canonical_chunk_ids": ["7bcdd909a9a4d4ac"],
            "expected_sources": [
                {"source_file": "4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf", "pages": [18], "section_title": "RMSE and MAE"}
            ],
            "notes": "Exact MAE formula: mean(|e_t|)."
        },
        {
            "query_id": "q02",
            "query": "What is the Simple Exponential Smoothing (SES) level updating formula?",
            "category": "formulas_and_notation",
            "difficulty": "easy",
            "relevant_records": {
                "1a71a9327db35525": 3,  # THE SES EQUATION
            },
            "canonical_chunk_ids": ["1a71a9327db35525"],
            "expected_sources": [
                {"source_file": "4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf", "pages": [7], "section_title": "THE SES EQUATION"}
            ],
            "notes": "Formula S_t = alpha * Y_t + (1-alpha) * S_{t-1}."
        },
        {
            "query_id": "q03",
            "query": "What is the mathematical equation for multiplicative time series decomposition?",
            "category": "formulas_and_notation",
            "difficulty": "easy",
            "relevant_records": {
                "4a6dee1879b3edbb": 3,  # WHEN Is Multiplicative Suitable? Y_t = T_t * S_t * I_t
            },
            "canonical_chunk_ids": ["4a6dee1879b3edbb"],
            "expected_sources": [
                {"source_file": "5. DS5104 Lecture5 Trend_Seasonal_Decomposition.pdf", "pages": [8], "section_title": "WHEN Is Multiplicative Suitable?"}
            ],
            "notes": "Formula Y_t = T_t * S_t * I_t."
        },
        {
            "query_id": "q04",
            "query": "What is the formula for the approximate significance bounds in an ACF plot?",
            "category": "formulas_and_notation",
            "difficulty": "easy",
            "relevant_records": {
                "d46d67f0d189e61d": 3,  # Significance Bounds in ACF: +- 1.96 / sqrt(n)
            },
            "canonical_chunk_ids": ["d46d67f0d189e61d"],
            "expected_sources": [
                {"source_file": "7. DS5104 Lecture7 Autocorrelation ACF.pdf", "pages": [17], "section_title": "Significance Bounds in ACF"}
            ],
            "notes": "Formula +- 1.96 / sqrt(n)."
        },
        {
            "query_id": "q05",
            "query": "What is the algebraic definition of first-order differencing for a time series?",
            "category": "formulas_and_notation",
            "difficulty": "easy",
            "relevant_records": {
                "3dc6711eb5d8651c": 3,  # First differencing definition: \nabla Y_t = Y_t - Y_{t-1}
            },
            "canonical_chunk_ids": ["3dc6711eb5d8651c"],
            "expected_sources": [
                {"source_file": "6. DS5104 Lecture6 Stationarity_and_Differencing.pdf", "pages": [13], "section_title": "HOW Do We Remove a Changing Level?"}
            ],
            "notes": "Formula nabla Y_t = Y_t - Y_{t-1}."
        },

        # =========================================================================
        # 2. Definitions and Terminology (5 questions)
        # =========================================================================
        {
            "query_id": "q06",
            "query": "What are the core mathematical properties that define a stationary time series?",
            "category": "definitions_and_terminology",
            "difficulty": "medium",
            "relevant_records": {
                "4178df85f5a5aeab": 3,  # WHY Do We Need Stationarity?
            },
            "canonical_chunk_ids": ["4178df85f5a5aeab"],
            "expected_sources": [
                {"source_file": "6. DS5104 Lecture6 Stationarity_and_Differencing.pdf", "pages": [4], "section_title": "WHY Do We Need Stationarity?"}
            ],
            "notes": "Constant mean, constant variance, covariance depending only on lag."
        },
        {
            "query_id": "q07",
            "query": "What is the definition of the Autocorrelation Function (ACF)?",
            "category": "definitions_and_terminology",
            "difficulty": "medium",
            "relevant_records": {
                "cde2dd8b4d4c7f79": 3,  # WHY Not Use Ordinary Correlation Only? (Autocorrelation compares series with its own past)
                "c37dd8d8bf6f26a8": 3,  # Sample Autocorrelation formula: r_k = \sum (Y_t - \bar{Y})(Y_{t-k} - \bar{Y}) / \sum (Y_t - \bar{Y})^2
            },
            "canonical_chunk_ids": ["cde2dd8b4d4c7f79", "c37dd8d8bf6f26a8"],
            "expected_sources": [
                {"source_file": "7. DS5104 Lecture7 Autocorrelation ACF.pdf", "pages": [4, 10], "section_title": "WHY Not Use Ordinary Correlation Only?"}
            ],
            "notes": "Correlation of a series with its own lagged values."
        },
        {
            "query_id": "q08",
            "query": "What is white noise in time series analysis and what are its properties?",
            "category": "definitions_and_terminology",
            "difficulty": "medium",
            "relevant_records": {
                "25009a64c356c405": 3,  # Interpretation: White Noise (No strong repeated structure, inside bounds)
                "bb649a69d546b686": 2,  # Pattern 1: White-Noise-Like Series
            },
            "canonical_chunk_ids": ["25009a64c356c405", "bb649a69d546b686"],
            "expected_sources": [
                {"source_file": "7. DS5104 Lecture7 Autocorrelation ACF.pdf", "pages": [18, 19], "section_title": "Interpretation: White Noise"}
            ],
            "notes": "Zero autocorrelation at all non-zero lags, constant mean, constant variance."
        },
        {
            "query_id": "q09",
            "query": "What is a Centered Moving Average (CMA) and why is it constructed?",
            "category": "definitions_and_terminology",
            "difficulty": "medium",
            "relevant_records": {
                "3cba47c636982e5e": 3,  # Centered Moving Average: Why Center?
            },
            "canonical_chunk_ids": ["3cba47c636982e5e"],
            "expected_sources": [
                {"source_file": "5. DS5104 Lecture5 Trend_Seasonal_Decomposition.pdf", "pages": [13], "section_title": "Centered Moving Average: Why Center?"}
            ],
            "notes": "Average of two consecutive even moving averages to align trend estimates with integer time points."
        },
        {
            "query_id": "q10",
            "query": "What are the four classical components of a time series?",
            "category": "definitions_and_terminology",
            "difficulty": "easy",
            "relevant_records": {
                "455e49bf5209a328": 3,  # WHAT Are the Components? Trend (T), Seasonal (S), Cycle (C), Irregular (I)
                "1a03a7138dfb0836": 2,  # Real-world component examples
            },
            "canonical_chunk_ids": ["455e49bf5209a328", "1a03a7138dfb0836"],
            "expected_sources": [
                {"source_file": "5. DS5104 Lecture5 Trend_Seasonal_Decomposition.pdf", "pages": [5], "section_title": "WHAT Are the Components?"}
            ],
            "notes": "Trend (T), Seasonality (S), Cycle (C), and Irregular/Random (I)."
        },

        # =========================================================================
        # 3. Conceptual Explanations (5 questions)
        # =========================================================================
        {
            "query_id": "q11",
            "query": "When is multiplicative decomposition appropriate compared to additive decomposition?",
            "category": "conceptual_explanations",
            "difficulty": "medium",
            "relevant_records": {
                "4a6dee1879b3edbb": 3,  # WHEN Is Multiplicative Suitable?
            },
            "canonical_chunk_ids": ["4a6dee1879b3edbb"],
            "expected_sources": [
                {"source_file": "5. DS5104 Lecture5 Trend_Seasonal_Decomposition.pdf", "pages": [8], "section_title": "WHEN Is Multiplicative Suitable?"}
            ],
            "notes": "When seasonal amplitude increases proportionally with the trend level."
        },
        {
            "query_id": "q12",
            "query": "Why does an even-period moving average fall between time periods and require centering?",
            "category": "conceptual_explanations",
            "difficulty": "medium",
            "relevant_records": {
                "3cba47c636982e5e": 3,  # Centered Moving Average: Why Center?
            },
            "canonical_chunk_ids": ["3cba47c636982e5e"],
            "expected_sources": [
                {"source_file": "5. DS5104 Lecture5 Trend_Seasonal_Decomposition.pdf", "pages": [13], "section_title": "Centered Moving Average: Why Center?"}
            ],
            "notes": "An even period average has no single integer midpoint, falling between time t and t+1."
        },
        {
            "query_id": "q13",
            "query": "What does the smoothing parameter alpha control in Simple Exponential Smoothing?",
            "category": "conceptual_explanations",
            "difficulty": "medium",
            "relevant_records": {
                "1a71a9327db35525": 3,  # THE SES EQUATION
            },
            "canonical_chunk_ids": ["1a71a9327db35525"],
            "expected_sources": [
                {"source_file": "4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf", "pages": [7], "section_title": "THE SES EQUATION"}
            ],
            "notes": "Controls balance between responsiveness to recent observation vs stability from past smoothed level."
        },
        {
            "query_id": "q14",
            "query": "Why do many time series forecasting models require the data to be stationary before model fitting?",
            "category": "conceptual_explanations",
            "difficulty": "medium",
            "relevant_records": {
                "4178df85f5a5aeab": 3,  # WHY Do We Need Stationarity?
            },
            "canonical_chunk_ids": ["4178df85f5a5aeab"],
            "expected_sources": [
                {"source_file": "6. DS5104 Lecture6 Stationarity_and_Differencing.pdf", "pages": [4], "section_title": "WHY Do We Need Stationarity?"}
            ],
            "notes": "Non-stationary patterns change over time, making future projections from historical statistics unreliable."
        },
        {
            "query_id": "q15",
            "query": "How does the autocorrelation function (ACF) behave for a series with strong trend versus pure white noise?",
            "category": "conceptual_explanations",
            "difficulty": "medium",
            "relevant_records": {
                "a0b64f8cdb5c6bdf": 3,  # Activity 3 • Match the Case
                "d46d67f0d189e61d": 2,  # Significance bounds
            },
            "canonical_chunk_ids": ["a0b64f8cdb5c6bdf", "d46d67f0d189e61d"],
            "expected_sources": [
                {"source_file": "7. DS5104 Lecture7 Autocorrelation ACF.pdf", "pages": [28], "section_title": "Activity 3 • Match the Case"}
            ],
            "notes": "Trend series has slow linear decay in ACF; white noise has virtually all bars within bounds."
        },

        # =========================================================================
        # 4. Multi-step Calculations and Workspaces (5 questions)
        # =========================================================================
        {
            "query_id": "q16",
            "query": "How is a Centered Moving Average calculated step-by-step for quarterly data in the decomposition workspace?",
            "category": "calculations_and_workspaces",
            "difficulty": "hard",
            "relevant_records": {
                "e9285418782c3cf7#frag_0": 3,  # CMA Workspace Step 1
                "e9285418782c3cf7#frag_1": 3,  # CMA Workspace Step 2
                "e9285418782c3cf7#frag_2": 2,  # CMA Workspace Step 3
            },
            "canonical_chunk_ids": ["e9285418782c3cf7"],
            "expected_sources": [
                {"source_file": "5. DS5104 Lecture5 Trend_Seasonal_Decomposition.pdf", "pages": [14, 15, 16], "section_title": "CMA Workspace"}
            ],
            "notes": "Step 1: 4-period moving sum/average. Step 2: 2-period centering average. Step 3: Detrending."
        },
        {
            "query_id": "q17",
            "query": "How are seasonal indices calculated and normalized in additive decomposition?",
            "category": "calculations_and_workspaces",
            "difficulty": "hard",
            "relevant_records": {
                "8472ec53de2f5e4f": 3,  # HOW Do We Estimate Seasonal Effects?
                "76d27801e42c68c7": 2,  # Seasonal Index: How to Read It
            },
            "canonical_chunk_ids": ["8472ec53de2f5e4f", "76d27801e42c68c7"],
            "expected_sources": [
                {"source_file": "5. DS5104 Lecture5 Trend_Seasonal_Decomposition.pdf", "pages": [19, 20], "section_title": "HOW Do We Estimate Seasonal Effects?"}
            ],
            "notes": "Detrended series values averaged for each season and normalized."
        },
        {
            "query_id": "q18",
            "query": "How do you calculate sequential Simple Exponential Smoothing forecasts given initial level S0 and alpha?",
            "category": "calculations_and_workspaces",
            "difficulty": "medium",
            "relevant_records": {
                "82cafee69eff2de1": 3,  # WORKED EXAMPLE — MONTHLY DEMAND
            },
            "canonical_chunk_ids": ["82cafee69eff2de1"],
            "expected_sources": [
                {"source_file": "4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf", "pages": [12], "section_title": "WORKED EXAMPLE — MONTHLY DEMAND"}
            ],
            "notes": "Step-by-step table computing S_t = alpha * Y_t + (1-alpha) * S_{t-1}."
        },
        {
            "query_id": "q19",
            "query": "How is Root Mean Squared Error (RMSE) calculated from observed and forecast values?",
            "category": "calculations_and_workspaces",
            "difficulty": "medium",
            "relevant_records": {
                "7bcdd909a9a4d4ac": 3,  # RMSE calculation
            },
            "canonical_chunk_ids": ["7bcdd909a9a4d4ac"],
            "expected_sources": [
                {"source_file": "4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf", "pages": [18], "section_title": "RMSE"}
            ],
            "notes": "Compute errors e_t, square them, take mean (MSE), then square root."
        },
        {
            "query_id": "q20",
            "query": "How is second-order differencing performed when first-order differencing leaves a trend?",
            "category": "calculations_and_workspaces",
            "difficulty": "hard",
            "relevant_records": {
                "1ce71e77c6493c61": 3,  # WHEN Do We Use Second Differencing?
            },
            "canonical_chunk_ids": ["1ce71e77c6493c61"],
            "expected_sources": [
                {"source_file": "6. DS5104 Lecture6 Stationarity_and_Differencing.pdf", "pages": [18], "section_title": "WHEN Do We Use Second Differencing?"}
            ],
            "notes": "Differentiating the first differences: nabla^2 Y_t = (Y_t - Y_{t-1}) - (Y_{t-1} - Y_{t-2})."
        },

        # =========================================================================
        # 5. Multi-page and Cross-lecture Questions (5 questions)
        # =========================================================================
        {
            "query_id": "q21",
            "query": "How does simple moving average smoothing differ from simple exponential smoothing in weighting historical data?",
            "category": "cross_lecture_and_multi_page",
            "difficulty": "hard",
            "relevant_records": {
                "c2e9d9ee70aa7089": 3,  # Moving average equal weighting (Lecture 3)
                "1a71a9327db35525": 3,  # SES exponential decay weighting (Lecture 4)
            },
            "canonical_chunk_ids": ["c2e9d9ee70aa7089", "1a71a9327db35525"],
            "expected_sources": [
                {"source_file": "3. DS5104 Lecture3 Smoothing_Methods_I.pdf", "pages": [6, 7], "section_title": "The main idea"},
                {"source_file": "4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf", "pages": [7], "section_title": "THE SES EQUATION"}
            ],
            "notes": "Moving averages assign equal 1/k weight; SES assigns geometrically decaying weights."
        },
        {
            "query_id": "q22",
            "query": "How is non-stationarity diagnosed using an ACF plot and resolved using differencing?",
            "category": "cross_lecture_and_multi_page",
            "difficulty": "hard",
            "relevant_records": {
                "4178df85f5a5aeab": 3,  # Stationarity and differencing (Lecture 6)
                "a0b64f8cdb5c6bdf": 3,  # ACF slow decay pattern (Lecture 7)
            },
            "canonical_chunk_ids": ["4178df85f5a5aeab", "a0b64f8cdb5c6bdf"],
            "expected_sources": [
                {"source_file": "6. DS5104 Lecture6 Stationarity_and_Differencing.pdf", "pages": [4], "section_title": "WHY Do We Need Stationarity?"},
                {"source_file": "7. DS5104 Lecture7 Autocorrelation ACF.pdf", "pages": [28], "section_title": "Activity 3 • Match the Case"}
            ],
            "notes": "Slow ACF decay indicates non-stationary trend, requiring first differencing."
        },
        {
            "query_id": "q23",
            "query": "How does seasonality manifest in decomposition models compared to periodic spikes in an ACF plot?",
            "category": "cross_lecture_and_multi_page",
            "difficulty": "hard",
            "relevant_records": {
                "4a6dee1879b3edbb": 3,  # Seasonal component in decomposition (Lecture 5)
                "a0b64f8cdb5c6bdf": 3,  # ACF seasonal spikes every m lags (Lecture 7)
            },
            "canonical_chunk_ids": ["4a6dee1879b3edbb", "a0b64f8cdb5c6bdf"],
            "expected_sources": [
                {"source_file": "5. DS5104 Lecture5 Trend_Seasonal_Decomposition.pdf", "pages": [8], "section_title": "WHEN Is Multiplicative Suitable?"},
                {"source_file": "7. DS5104 Lecture7 Autocorrelation ACF.pdf", "pages": [28], "section_title": "Activity 3 • Match the Case"}
            ],
            "notes": "Decomposition explicitly isolates S_t; ACF displays significant positive spikes at lags m, 2m, 3m."
        },
        {
            "query_id": "q24",
            "query": "How are forecast accuracy metrics used to select the optimal smoothing parameter alpha in SES?",
            "category": "cross_lecture_and_multi_page",
            "difficulty": "medium",
            "relevant_records": {
                "7bcdd909a9a4d4ac": 3,  # Accuracy metrics RMSE / MAE
                "1a71a9327db35525": 2,  # SES equation
            },
            "canonical_chunk_ids": ["7bcdd909a9a4d4ac", "1a71a9327db35525"],
            "expected_sources": [
                {"source_file": "4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf", "pages": [18, 22], "section_title": "RMSE"}
            ],
            "notes": "Select alpha that minimizes in-sample RMSE or MAE."
        },
        {
            "query_id": "q25",
            "query": "What defines the random irregular component in decomposition and how does it relate to white noise in ACF?",
            "category": "cross_lecture_and_multi_page",
            "difficulty": "hard",
            "relevant_records": {
                "3cba47c636982e5e": 2,  # Decomposition residual (Lecture 5)
                "25009a64c356c405": 3,  # Interpretation: White Noise (Lecture 7)
                "d46d67f0d189e61d": 2,  # ACF significance bounds (Lecture 7)
            },
            "canonical_chunk_ids": ["3cba47c636982e5e", "25009a64c356c405", "d46d67f0d189e61d"],
            "expected_sources": [
                {"source_file": "5. DS5104 Lecture5 Trend_Seasonal_Decomposition.pdf", "pages": [13], "section_title": "Centered Moving Average: Why Center?"},
                {"source_file": "7. DS5104 Lecture7 Autocorrelation ACF.pdf", "pages": [17, 19], "section_title": "Interpretation: White Noise"}
            ],
            "notes": "A well-fit model leaves irregular residuals that resemble uncorrelated white noise."
        },

        # =========================================================================
        # 6. Ambiguous and Terminology-Variant Questions (5 questions)
        # =========================================================================
        {
            "query_id": "q26",
            "query": "What does alpha represent in forecasting algorithms?",
            "category": "ambiguous_and_variants",
            "difficulty": "medium",
            "relevant_records": {
                "1a71a9327db35525": 3,  # SES equation
            },
            "canonical_chunk_ids": ["1a71a9327db35525"],
            "expected_sources": [
                {"source_file": "4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf", "pages": [7], "section_title": "THE SES EQUATION"}
            ],
            "notes": "Tests variant query 'alpha in forecasting algorithms' resolving to SES smoothing parameter."
        },
        {
            "query_id": "q27",
            "query": "Formula for error measurement using absolute values rather than squared differences",
            "category": "ambiguous_and_variants",
            "difficulty": "medium",
            "relevant_records": {
                "7bcdd909a9a4d4ac": 3,  # MAE vs MSE
            },
            "canonical_chunk_ids": ["7bcdd909a9a4d4ac"],
            "expected_sources": [
                {"source_file": "4. DS5104 Lecture4 Simple_Exponential_Smoothing.pdf", "pages": [18], "section_title": "RMSE"}
            ],
            "notes": "Tests descriptive conceptual variant without mentioning the acronym MAE."
        },
        {
            "query_id": "q28",
            "query": "How do you transform a time series when the mean is drifting over time?",
            "category": "ambiguous_and_variants",
            "difficulty": "medium",
            "relevant_records": {
                "4178df85f5a5aeab": 3,  # WHY Do We Need Stationarity? (Lecture 6)
                "3dc6711eb5d8651c": 3,  # HOW Do We Remove a Changing Level? First difference (Lecture 6)
            },
            "canonical_chunk_ids": ["4178df85f5a5aeab", "3dc6711eb5d8651c"],
            "expected_sources": [
                {"source_file": "6. DS5104 Lecture6 Stationarity_and_Differencing.pdf", "pages": [4, 13], "section_title": "HOW Do We Remove a Changing Level?"}
            ],
            "notes": "Tests non-standard phrasing for trend non-stationarity resolving to differencing."
        },
        {
            "query_id": "q29",
            "query": "Why average two neighboring 4-period moving averages?",
            "category": "ambiguous_and_variants",
            "difficulty": "medium",
            "relevant_records": {
                "3cba47c636982e5e": 3,  # Centered Moving Average: Why Center?
            },
            "canonical_chunk_ids": ["3cba47c636982e5e"],
            "expected_sources": [
                {"source_file": "5. DS5104 Lecture5 Trend_Seasonal_Decomposition.pdf", "pages": [13], "section_title": "Centered Moving Average: Why Center?"}
            ],
            "notes": "Informal student phrasing for 2x4 CMA centering."
        },
        {
            "query_id": "q30",
            "query": "When are correlation bars statistically significant in time series?",
            "category": "ambiguous_and_variants",
            "difficulty": "medium",
            "relevant_records": {
                "d46d67f0d189e61d": 3,  # Significance bounds in ACF
            },
            "canonical_chunk_ids": ["d46d67f0d189e61d"],
            "expected_sources": [
                {"source_file": "7. DS5104 Lecture7 Autocorrelation ACF.pdf", "pages": [17], "section_title": "Significance Bounds in ACF"}
            ],
            "notes": "Informal phrasing 'correlation bars' for ACF spikes outside +- 1.96 / sqrt(n)."
        },
    ]

    # Verify that all chunk IDs exist in the 238 records inventory
    all_record_ids = {c["record_id"] for c in inventory}
    all_parent_ids = {c["chunk_id"] for c in inventory}

    missing_ids = []
    for q in queries:
        for cid in q["relevant_records"].keys():
            if cid not in all_record_ids and cid not in all_parent_ids:
                missing_ids.append((q["query_id"], cid))

    if missing_ids:
        print(f"ERROR: Missing chunk IDs in benchmark: {missing_ids}")
        return

    benchmark_path = Path("data/evaluation/retrieval_benchmark_v1.json")
    with open(benchmark_path, "w", encoding="utf-8") as out:
        json.dump(queries, out, indent=2)

    print(f"\nSUCCESS: Created benchmark with {len(queries)} questions in {benchmark_path}")


if __name__ == "__main__":
    main()
