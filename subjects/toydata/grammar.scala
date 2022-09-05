/*
 * This file contains the grammar for the example.
 * This is a scala-based dsl, also used within tribble (https://github.com/havrikov/tribble).
 * Please have a look at their documentation to understand the format.
 */
Grammar(
  'EXPRESSION := ("[" ~ "[" ~ 'bool0 ~ "," ~ 'bool1 ~ "," ~ 'bool2 ~ "," ~ 'bool3 ~ "," ~ 'bool4 ~ "," ~ 'bool5 ~ "," ~ 'bool6 ~ "," ~ 'bool7 ~ "," ~ 'bool8 ~ "," ~ 'bool9 ~ "]" ~ "]\n"),
  'bool0 := "0" | "1",
  'bool1 := "0" | "1",
  'bool2 := "0" | "1",
  'bool3 := "0" | "1",
  'bool4 := "0" | "1",
  'bool5 := "0" | "1",
  'bool6 := "0" | "1",
  'bool7 := "0" | "1",
  'bool8 := "0" | "1",
  'bool9 := "0" | "1"
)

