-- creates a safely divide function

DELIMITER $$

CREATE FUNCTION SafeDiv(a INT, b INT)

RETURNS FLOAT

BEGIN
	DECLARE result FLOAT;
	SET result = 0.0;
	IF b = 0 THEN
	   SET result = 0;
	ELSE
	   SET result = a / b;
	END IF;
	
	RETURN result;
END $$

DELIMITER ;