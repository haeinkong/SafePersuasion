def make_fewshot_prompt(df, exclude_idx):
    
    fewshot=""
    pool_df = df.drop(index=exclude_idx)

    for label in pool_df["second_label"].unique():
        class_df = pool_df[pool_df["second_label"] == label]
        samples = class_df.sample(n=1)
        
        for _, row in samples.iterrows():
            fewshot += f"Text: {row['text']}\nAnswer: {label}\n\n"
                    
    return fewshot
