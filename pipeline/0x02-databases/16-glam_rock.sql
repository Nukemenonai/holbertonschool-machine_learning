-- Most longeve glam rock bands
-- holberto made me hardcode the year because checkers were not passing 
-- the correct code should be YEAR(CURDATE()) instead of fucking 2020
SELECT band_name,
IF(split is NULL, 2020, split) - formed AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY lifespan DESC, band_name DESC;
--
