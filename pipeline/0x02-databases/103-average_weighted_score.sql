-- computes and stores the average weighted store for a student

DELIMITER $$

CREATE PROCEDURE ComputeAverageWeightedScoreForUser (
    IN new_user_id INTEGER)
BEGIN
     UPDATE users SET average_score=(SELECT SUM(weight * score) / SUM(weight)
     	    	      		     FROM projects
				     JOIN corrections ON corrections.project_id = projects.id
				     WHERE corrections.user_id = new_user_id)
     WHERE users.id = new_user_id;
END $$

DELIMITER ;