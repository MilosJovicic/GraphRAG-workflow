WITH [t IN $terms | toLower(t)] AS lower_terms
MATCH (n:SettingKey)
WITH n, lower_terms, toLower(n.key) AS lkey
WITH n, lkey,
  CASE
    WHEN ANY(t IN lower_terms WHERE lkey = t) THEN 3.0
    WHEN ANY(t IN lower_terms WHERE size(t) >= 4 AND lkey STARTS WITH t) THEN 2.0
    WHEN ANY(t IN lower_terms WHERE size(lkey) >= 4 AND t STARTS WITH lkey) THEN 1.0
    ELSE 0.0
  END AS relevance
WHERE relevance > 0
OPTIONAL MATCH (def:Section)-[:DEFINES]->(n)
WITH n, relevance, collect(DISTINCT def) AS defs
WITH n, relevance, defs[0] AS def_section
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(def_section)
RETURN
  COALESCE(n.id, 'SettingKey:' + n.key)                             AS node_id,
  'SettingKey'                                                      AS node_label,
  COALESCE(n.indexed_text, n.context, n.description, n.key)         AS indexed_text,
  COALESCE(n.description, n.context, n.key)                         AS raw_text,
  p.url                                                             AS url,
  def_section.anchor                                                AS anchor,
  def_section.breadcrumb                                            AS breadcrumb,
  COALESCE(n.title, n.key)                                          AS title,
  relevance                                                         AS bm25_score
ORDER BY relevance DESC, n.key
LIMIT $limit
