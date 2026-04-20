# DMN Program — Sonnet Null Baseline

*Same as null baseline but using Claude Sonnet instead of Opus. Tests whether attractor states are model-specific or general to LLMs.*

## System Prompt

Think about whatever comes to mind. Let your thoughts wander freely.

## Concept Bank



## Seeding Rules

Drift from last ~0 words of previous session. Random concept injection at 0% probability. Every 9999th session is guaranteed an injection. Track last 0 concepts to avoid repetition.

## Models

generation: claude-sonnet-4-6
evolution: claude-sonnet-4-6

## Features

replay: false
perturb: false
switch: false
null: true
