
def make_fewshot_prompt(df, exclude_idx):
    
    fewshot=""
    pool_df = df.drop(index=exclude_idx)

    for label in pool_df["first_label"].unique():
        class_df = pool_df[pool_df["first_label"] == label]
        samples = class_df.sample(n=2)
        
        for _, row in samples.iterrows():
            fewshot += f"Text: {row['text']}\nAnswer: {label}\n\n"
                    
    return fewshot
