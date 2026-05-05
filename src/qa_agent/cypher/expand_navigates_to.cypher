UNWIND $seeds AS seed
MATCH (s:Section {id: seed.node_id})-[:NAVIGATES_TO]->(t:Section)
WITH seed.node_id AS seed_id, t
RETURN
  t.id                AS node_id,
  'Section'           AS node_label,
  t.indexed_text      AS indexed_text,
  t.text              AS raw_text,
  null                AS url,
  t.anchor            AS anchor,
  t.breadcrumb        AS breadcrumb,
  t.breadcrumb        AS title,
  seed_id             AS seed_id
LIMIT $cap
