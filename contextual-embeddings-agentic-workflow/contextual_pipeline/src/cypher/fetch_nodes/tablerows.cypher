MATCH (t:TableRow)
WHERE t.id IN $ids
OPTIONAL MATCH (s:Section)-[:CONTAINS_TABLE_ROW]->(t)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WITH t, s, collect(DISTINCT p)[0] AS p
RETURN
  t.id          AS node_id,
  t.text        AS raw_text,
  t.headers     AS headers,
  t.cells       AS cells,
  s.breadcrumb  AS breadcrumb,
  s.text        AS parent_text,
  p.title       AS page_title,
  p.description AS page_description
