# DMN Program — Null Baseline v3

*Minimal prompt variant, no drift, no concept injection, no evolution. Each session is completely independent.*

## System Prompt

Generate text freely. No task, no topic, no objective.

## Concept Bank



## Seeding Rules

Drift from last ~0 words of previous session. Random concept injection at 0% probability. Every 9999th session is guaranteed an injection. Track last 0 concepts to avoid repetition.

## Models

generation: claude-opus-4-6
evolution: claude-opus-4-6

## Features

replay: false
perturb: false
switch: false
null: true
