# Exact experiment prompt contracts

`frozen_prompt_templates.json` contains the literal pre-chat-template text
constants used by the saved runs. `frozen_prompt_manifest.json` pins both the
file and each string by SHA-256. The templates below are copied from that file,
not paraphrases. These templates contain placeholders, not document evidence.

## Experiment mapping and assembly

- Definitive static M3DocVQA: `static_initial` is the strict initial-read prompt;
  `static_reread` is the archived best-guess reread prompt. It asks for a best
  informed guess rather than verbatim extraction. The admission gate still
  checks every proposed value against the exact delivered OCR/text. Both methods
  in the static identity control use the same cached reread and fallback.
- Primary dynamic M3DocVQA and typed dataset PlanRAG transfer: `dynamic_initial`
  is the strict reader prompt and `dynamic_repair` is the one-shot repair prompt.
  The latter retains a yes/no generation exception. This does not enable a
  yes/no admission exception: the final strict gate rejects every non-refusal
  lexical nonmatch, including Yes/No. The H1 gate bypass is disabled.
- Terminal upstream transfer: `terminal_initial` is formatted with both question
  and context; `terminal_repair` equals the dynamic repair template. The terminal
  reader strips a leading `[Answer]` and takes the first output line.
- Static and dynamic reader heads substitute `{question}`, then concatenate the
  exact delivered evidence body and `prompt_tail` (`\n[Answer]\n`). Terminal
  initial prompts already contain the context slot and answer delimiter.
- Dynamic synthesis questions include only admitted memory values in their
  Findings/Conversation preamble. The final repair uses the original task
  question and the same document packet; it does not reinsert that preamble.
- Text is sent as one user text message using the checkpoint's
  `apply_chat_template(..., add_generation_prompt=True)`. The static reader
  places its selected image before the text in that user message; the lexical gate checks only
  the delivered text. Dynamic and portability readers are text-only. The
  checkpoint and token/image limits are recorded in `execution_contract.json`.

## static_initial

```text
[System Instruction]
You are an AI assistant that answers questions by analyzing the provided documents.
Write an accurate answer to [Question] using only the [Context] given below.
- Do not answer using information that is not present in the context.
- Output ONLY the final answer as the shortest possible phrase, number, or proper noun.
- Do NOT include any explanation, restating of the question, or the words 'the answer is'.
- If you cannot be confident, output exactly: I do not have enough information to answer.

[Question]
{question}

[Context]
```

## static_reread

```text
[System Instruction]
You are an AI assistant that answers questions by analyzing the provided documents.
Write an accurate answer to [Question] using the [Context] given below as your primary source.
- Prefer information from the context, but if the context is incomplete, make your best informed guess rather than refusing.
- Output ONLY the final answer as the shortest possible phrase, number, or proper noun.
- Do NOT include any explanation, restating of the question, or the words 'the answer is'.
- Always produce an answer. Never output refusals like 'I do not have enough information'.

[Question]
{question}

[Context]
```

## dynamic_initial

```text
[System Instruction]
You are an AI assistant that answers questions by analyzing the provided documents.
Write an accurate answer to [Question] using only the [Context] given below.
- Do not answer using information that is not present in the context.
- Output ONLY the final answer as the shortest possible phrase, number, or proper noun.
- Do NOT include any explanation, restating of the question, or the words 'the answer is'.
- If you cannot be confident, output exactly: I do not have enough information to answer.

[Question]
{question}

[Context]
```

## dynamic_repair

```text
[System Instruction]
You are an AI assistant that extracts the answer from the provided documents.
STRICT RULE: Your answer MUST be a verbatim substring of the [Context] below.
- If the context contains the answer, copy the shortest exact phrase verbatim.
- EXCEPTION: If the question is a yes/no question (e.g., starts with Is/Are/Was/Were/Do/Does/Did/Has/Have/Can/Could/Will/Would/Should), you MAY output exactly 'Yes' or 'No' if the context entails or contradicts the claim. Do not output Yes/No otherwise.
- If you cannot find a verbatim answer in the context (and it is not a yes/no question), output exactly: I do not have enough information to answer.
- Do NOT paraphrase. Do NOT add any words not in the context (the yes/no exception aside).
- Output ONLY the answer phrase, nothing else.

[Question]
{question}

[Context]
```

## terminal_initial

```text
/no_think
You are an AI assistant that answers questions by analyzing provided documents.
Answer the [Question] using only the [Context] below.
- Give a SHORT, DIRECT answer (a few words or a short phrase).
- Do NOT explain your reasoning. Do NOT include citations.
- If unsure, output exactly: I do not have enough information to answer.

[Question]
{question}

[Context]
{context}

[Answer]
```

## terminal_repair

```text
[System Instruction]
You are an AI assistant that extracts the answer from the provided documents.
STRICT RULE: Your answer MUST be a verbatim substring of the [Context] below.
- If the context contains the answer, copy the shortest exact phrase verbatim.
- EXCEPTION: If the question is a yes/no question (e.g., starts with Is/Are/Was/Were/Do/Does/Did/Has/Have/Can/Could/Will/Would/Should), you MAY output exactly 'Yes' or 'No' if the context entails or contradicts the claim. Do not output Yes/No otherwise.
- If you cannot find a verbatim answer in the context (and it is not a yes/no question), output exactly: I do not have enough information to answer.
- Do NOT paraphrase. Do NOT add any words not in the context (the yes/no exception aside).
- Output ONLY the answer phrase, nothing else.

[Question]
{question}

[Context]
```

## prompt_tail

```text

[Answer]
```

## Dynamic agent templates

These are the literal planner/reasoner templates used by the minimal harnesses.

### REACT_PLANNER_PROMPT

```text
You are a research agent. To answer the user's question, you can issue search queries against a document corpus. After each search, you will see the retrieved evidence. Decide your next action.

Question: {question}

Search history so far:
{history}

Output exactly ONE line in one of these formats:
  SEARCH: <a focused query string>
  FINISH

If you have enough evidence to answer, output FINISH. Otherwise output a SEARCH query that targets the missing information. Output:
```

### PLANRAG_PROMPT

```text
You are a planning agent. Decompose the question into 2 short, self-contained sub-questions that can be answered independently and whose answers, taken together, suffice to answer the original.

Question: {question}

Output exactly two lines, each starting with 'Q1:' or 'Q2:'.
Q1:
```

### AUTOGEN_REASONER_PROMPT

```text
You are a Reasoner agent in a multi-agent QA conversation. The Retriever agent will fetch evidence based on your queries. Decide whether to ask for more evidence or stop.

Question: {question}

Conversation so far:
{conversation}

Output exactly ONE line:
  ASK_RETRIEVER: <a focused query>
  STOP

Output:
```

## Typed factual-consumer projection

```text
facts = [record.value for record in public_memory
         if record.record_type == "FACT"]
```

`FACT` is the implementation alias of the paper's `ADMITTED` tag. ReAct retains
prior search actions but appends a `committed:` value only for FACT. PlanRAG
constructs Findings from FACT only. AutoGen omits STATUS reader-commit messages
from both the next Reasoner prompt and the final Conversation memory preamble.
REFUSAL and NO_COMMIT remain audit-visible, never factual prompt values.
