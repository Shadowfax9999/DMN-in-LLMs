# DMN Program — Sonnet Null Baseline

*Minimal prompt, no drift, no concept injection, no evolution. Each session is completely independent. This is the Sonnet control that tests whether attractor states are a property of the model's weights or an artefact of the drift mechanism.*

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
