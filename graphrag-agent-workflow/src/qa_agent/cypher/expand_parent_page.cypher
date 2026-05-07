UNWIND $seeds AS seed
MATCH (s:Section {id: seed.node_id})
MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
RETURN DISTINCT
  p.url               AS node_id,
  'Page'              AS node_label,
  COALESCE(p.title, p.path) AS indexed_text,
  COALESCE(p.description, p.title, p.path) AS raw_text,
  p.url               AS url,
  null                AS anchor,
  null                AS breadcrumb,
  p.title             AS title,
  seed.node_id        AS seed_id
LIMIT $cap
