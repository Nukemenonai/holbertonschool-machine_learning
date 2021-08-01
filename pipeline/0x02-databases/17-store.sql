--creates a trigger that decreases quantityof an item after adding a new order

DELIMITER $$
CREATE TRIGGER items_updated
   AFTER INSERT
   ON orders FOR EACH ROW
BEGIN
   UPDATE items SET quantity = quantity - NEW.number
   WHERE items.name = NEW.item_name;
END $$

DELIMITER ;