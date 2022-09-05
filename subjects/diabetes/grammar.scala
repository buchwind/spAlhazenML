/*
 * This file contains the grammar for the example.
 * This is a scala-based dsl, also used within tribble (https://github.com/havrikov/tribble).
 * Please have a look at their documentation to understand the format.
 */
Grammar(
  'EXPRESSION := ("[" ~ "[" ~ 'Age ~ "," ~ 'Gender ~ "," ~ 'Polyuria ~ "," ~ 'Polydipsia ~ "," ~ 'sudden_weight_loss ~ "," ~ 'weakness ~ "," ~ 'Polyphagia ~ "," ~ 'Genital_thrush ~ "," ~ 'visual_blurring ~ "," ~ 'Itching ~ "," ~ 'Irritability ~ "," ~ 'delayed_healing ~ "," ~ 'partial_paresis ~ "," ~ 'muscle_stiffness ~ "," ~ 'Alopecia ~ "," ~ 'Obesity ~ "]" ~ "]\n"),
  'Age := "[1-9]".regex ~ "[0-9]".regex,  
  'Gender := "0" | "1",
  'Polyuria := "0" | "1",
  'Polydipsia := "0" | "1",
  'sudden_weight_loss := "0" | "1",
  'weakness := "0" | "1",
  'Polyphagia := "0" | "1",
  'Genital_thrush := "0" | "1",
  'visual_blurring := "0" | "1",
  'Itching := "0" | "1",
  'Irritability := "0" | "1",
  'delayed_healing := "0" | "1",
  'partial_paresis := "0" | "1",
  'muscle_stiffness := "0" | "1",
  'Alopecia := "0" | "1",
  'Obesity := "0" | "1"
)

