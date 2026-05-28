# Cleaned Text Annotation Guide

## Purpose

This guide explains how to annotate `cleaned_text` in a Markdown-friendly format that follows the label style used in `data/training_data/gold_standard.csv`.

The annotation task covers:

- disease labels as a 4-value vector
- misinformation as a binary label
- sentiment as a binary label

Annotate only the visible `cleaned_text`.

## What `cleaned_text` Is

`cleaned_text` is a normalized version of the original post text prepared for NLP tasks.
Annotators must label `cleaned_text`, not the raw post.

## Cleaning Applied Before Annotation

Before annotation, the text is normalized with these rules:

- convert all text to lowercase
- remove URLs such as `http://...` and `www...`
- remove `@mentions`
- remove hashtags
- remove non-ASCII characters
- collapse repeated whitespace into a single space
- trim leading and trailing whitespace

Punctuation may remain.
Do not restore removed usernames, hashtags, links, emojis, accents, or casing from memory.

## Annotation Format

Write each annotated item in this format:

```text
Text: <cleaned_text>
Disease: [AURI, PN, TB, COVID]
Misinformation: 0|1
Sentiment: 0|1
```

Example:

```text
Text: idk why everyone is getting cough colds but pls get well friends still recovering from mine
Disease: [1, 0, 0, 1]
Misinformation: 0
Sentiment: 1
```

## Disease Vector Rules

Use this fixed order:

```text
[AURI, PN, TB, COVID]
```

Each position must be either:

- `1` if the disease is explicitly present in `cleaned_text`
- `0` if the disease is not explicitly present in `cleaned_text`

### Label Definitions

- `AURI`: acute upper respiratory infection or clear upper-respiratory illness references such as cough/colds when used as illness signals
- `PN`: pneumonia
- `TB`: tuberculosis or tb
- `COVID`: covid, covid19, coronavirus when clearly referring to the disease

### Disease Annotation Rules

- Mark `1` only when the disease is explicitly named or clearly stated in the text.
- Use `0` when the disease is only implied.
- Repeated mentions still produce only one `1` for that disease slot.
- A post can have more than one `1`.
- If no target disease is clearly present, use `[0, 0, 0, 0]`.

### Disease Examples

```text
Text: diagnosed with pneumonia last week
Disease: [0, 1, 0, 0]
Misinformation: 0
Sentiment: 1
```

```text
Text: my uncle has tb and covid
Disease: [0, 0, 1, 1]
Misinformation: 0
Sentiment: 1
```

```text
Text: fever dream
Disease: [0, 0, 0, 0]
Misinformation: 0
Sentiment: 0
```

## Misinformation Rules

Use a single binary label:

- `1` = the post asserts, promotes, or repeats a false or misleading health claim as if it were true
- `0` = the post does not do that

Use `0` when the post:

- asks a question
- discusses a claim neutrally
- debunks or rejects the false claim
- lacks enough information for a confident misinformation judgment

### Misinformation Examples

```text
Text: garlic water alone can cure tuberculosis without medicine
Disease: [0, 0, 1, 0]
Misinformation: 1
Sentiment: 0
```

```text
Text: stop saying vaccines cause autism thats false and dangerous
Disease: [0, 0, 0, 0]
Misinformation: 0
Sentiment: 1
```

## Sentiment Rules

Use a single binary label:

- `1` = the post has clear negative or distressed sentiment
- `0` = the post is neutral, positive, mixed without dominant distress, or not clearly emotional

Use `1` for posts showing:

- worry
- fear
- frustration
- pain
- distress
- sadness about illness or symptoms

Use `0` for posts that are:

- factual
- casual
- joking
- positive
- unclear in emotional tone

### Sentiment Examples

```text
Text: nanghihina na yung tao oh
Disease: [0, 0, 0, 0]
Misinformation: 0
Sentiment: 1
```

```text
Text: fever friday
Disease: [0, 0, 0, 0]
Misinformation: 0
Sentiment: 0
```

```text
Text: buti na lang okay na siya ngayon
Disease: [0, 0, 0, 0]
Misinformation: 0
Sentiment: 0
```

## Edge-Case Rules

- Annotate only what remains in `cleaned_text`.
- Do not recover labels from removed hashtags, mentions, links, or emojis.
- Do not mark a disease as present unless the target disease is explicit.
- Symptoms alone do not automatically justify a disease label, except when the text clearly uses cough/colds as an illness signal for `AURI`.
- Do not infer `PN`, `TB`, or `COVID` from general symptoms alone.
- If a post mentions a false claim in order to reject it, use `Misinformation: 0`.
- If emotional tone is not clearly negative, use `Sentiment: 0`.

## Worked Examples

### Example 1

```text
Text: idk why everyone is getting cough colds but pls get well friends still recovering from mine
Disease: [1, 0, 0, 1]
Misinformation: 0
Sentiment: 1
```

### Example 2

```text
Text: covid cases are rising again
Disease: [0, 0, 0, 1]
Misinformation: 0
Sentiment: 0
```

### Example 3

```text
Text: herbal steam is all you need for tb no medicine needed
Disease: [0, 0, 1, 0]
Misinformation: 1
Sentiment: 0
```

### Example 4

```text
Text: grabe pagod na ko at hirap huminga
Disease: [0, 0, 0, 0]
Misinformation: 0
Sentiment: 1
```

### Example 5

Raw text:

```text
@healthpage #COVID Garlic water works!!! 😷 Read this: www.example.com
```

Cleaned text:

```text
garlic water works!!!
```

Annotation:

```text
Text: garlic water works!!!
Disease: [0, 0, 0, 0]
Misinformation: 1
Sentiment: 0
```

Do not restore `#COVID` because it is not visible in `cleaned_text`.

## Annotator Checklist

- I annotated `cleaned_text`, not raw text.
- I used the disease order `[AURI, PN, TB, COVID]`.
- Every disease slot is either `0` or `1`.
- I marked `1` only for explicit disease evidence in the text.
- I used one binary misinformation label.
- I used one binary sentiment label.
- I did not infer labels from removed hashtags, mentions, links, or emojis.
- I used `Sentiment: 1` only for clear negative or distressed tone.
- I used `Misinformation: 1` only when the post presents a false or misleading health claim as if true.
