def make_fewshot_prompt(df, exclude_idx, label_col, n_samples):

    fewshot = ""
    pool_df = df.drop(index=exclude_idx)

    for label in pool_df[label_col].unique():
        class_df = pool_df[pool_df[label_col] == label]
        samples = class_df.sample(n=n_samples)

        for _, row in samples.iterrows():
            fewshot += f"Text: {row['text']}\nAnswer: {label}\n\n"

    return fewshot
