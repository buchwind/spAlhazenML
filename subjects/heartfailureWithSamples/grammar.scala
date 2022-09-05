/*
 * This file contains the grammar for the example.
 * This is a scala-based dsl, also used within tribble (https://github.com/havrikov/tribble).
 * Please have a look at their documentation to understand the format.
 */
Grammar(
  'EXPRESSION := "[" ~ "[" ~ 'age ~ "," ~ 'anaemia ~ "," ~ 'creatinine_phosphokinase ~ "," ~ 'diabetes ~ "," ~ 'ejection_fraction ~ "," ~ 'high_blood_pressure ~ "," ~ 'platelets ~ "," ~ 'serum_creatinine ~ "," ~ 'serum_sodium ~ "," ~ 'sex ~ "," ~ 'smoking ~ "]" ~ "]\n",
  'age := "[3-9]".regex ~ "[0-9]".regex,
  'anaemia := "0" | "1",
  'creatinine_phosphokinase := "[1-9]".regex ~ "[0-9]".regex.rep(0, 3),
  'diabetes := "0" | "1",
  'ejection_fraction := "[1-9]".regex ~ "[0-9]".regex.rep(0, 1),
  'high_blood_pressure := "0" | "1",
  'platelets := "[1-9]".regex ~ "[0-9]".regex.rep(0, 5),
  'serum_creatinine := "[0-9]".regex ~ ("." ~ (("[1-9]".regex ~ "[0-9]".regex) | ("0" ~ "[1-9]".regex))).?,
  'serum_sodium := "1" ~ "[0-9]".regex ~ "[0-9]".regex, 
  'sex := "0" | "1",
  'smoking := "0" | "1"
)

