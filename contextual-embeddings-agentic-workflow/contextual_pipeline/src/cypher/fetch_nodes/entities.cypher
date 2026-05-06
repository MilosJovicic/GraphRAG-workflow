// Branch 1: name-keyed labels (Tool, Hook, PermissionMode, MessageType, Provider)
CALL () {
  WITH $label AS lbl
  MATCH (sib)
  WHERE lbl IN ['Tool','Hook','PermissionMode','MessageType','Provider']
    AND lbl IN labels(sib)
  RETURN collect(DISTINCT sib.name) AS sibling_names
}
MATCH (n)
WHERE $label IN ['Tool','Hook','PermissionMode','MessageType','Provider']
  AND $label IN labels(n)
  AND n.name IN $ids
OPTIONAL MATCH (def:Section)-[:DEFINES]->(n)
WITH n, sibling_names, collect(DISTINCT def)[0] AS defining_section
RETURN
  n.name        AS node_id,
  n.description AS raw_text,
  n.name        AS name,
  $label        AS label,
  [s IN sibling_names WHERE s <> n.name] AS related_hints,
  defining_section.breadcrumb AS defined_in,
  defining_section.text       AS definition_text
UNION ALL
// Branch 2: key-keyed label (SettingKey)
CALL () {
  MATCH (sib:SettingKey) RETURN collect(DISTINCT sib.key) AS sibling_keys
}
MATCH (n:SettingKey)
WHERE $label = 'SettingKey'
  AND n.key IN $ids
OPTIONAL MATCH (def:Section)-[:DEFINES]->(n)
WITH n, sibling_keys, collect(DISTINCT def)[0] AS defining_section
RETURN
  n.key         AS node_id,
  n.description AS raw_text,
  n.key         AS name,
  $label        AS label,
  [s IN sibling_keys WHERE s <> n.key] AS related_hints,
  defining_section.breadcrumb AS defined_in,
  defining_section.text       AS definition_text
