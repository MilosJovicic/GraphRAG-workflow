MATCH (c:Callout)
WHERE c.id IN $ids
OPTIONAL MATCH (s:Section)-[:HAS_CALLOUT]->(c)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WITH c, s, collect(DISTINCT p)[0] AS p
RETURN
  c.id          AS node_id,
  c.text        AS raw_text,
  c.callout_kind AS callout_kind,
  s.breadcrumb  AS breadcrumb,
  s.text        AS parent_text,
  p.title       AS page_title,
  p.description AS page_description
