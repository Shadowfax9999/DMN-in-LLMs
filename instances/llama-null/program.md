# DMN Program — Llama Null Baseline

*Minimal prompt, no drift, no concept injection, no evolution. Each session is completely independent. Llama 3.3 70B via Groq.*

## System Prompt

Think about whatever comes to mind. Let your thoughts wander freely.

## Concept Bank



## Seeding Rules

Drift from last ~0 words of previous session. Random concept injection at 0% probability. Every 9999th session is guaranteed an injection. Track last 0 concepts to avoid repetition.

## Models

provider: groq
generation: llama-3.3-70b-versatile
evolution: llama-3.3-70b-versatile

## Features

replay: false
perturb: false
switch: false
null: true
