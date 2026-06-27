import type { StageConfig } from "@/lib/api"

export const SUPPORTED_DIRECTIONS = ["en-zh", "zh-en", "ja-zh"] as const
export type Direction = (typeof SUPPORTED_DIRECTIONS)[number]

export const DEFAULT_ASR_LANGUAGE = "en"
export const DEFAULT_TARGET_LANGUAGE = "zh"
export const DEFAULT_DIRECTION: Direction = "en-zh"

export function directionToLanguages(direction: Direction) {
  const [asr_language, target_language] = direction.split("-")
  return { asr_language, target_language }
}

export function directionFromConfig(config: Pick<StageConfig, "asr_language" | "target_language">): Direction {
  const direction = `${config.asr_language}-${config.target_language}`
  return SUPPORTED_DIRECTIONS.includes(direction as Direction) ? (direction as Direction) : DEFAULT_DIRECTION
}
