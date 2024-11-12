mean_field_term = inputs[0].PointData['mean_field_term']
entropy_term = inputs[0].PointData['entropy_term']


L1_term = inputs[1].PointData["L1_term"]
L2_term = inputs[1].PointData["L2_term"]
L3_term = inputs[1].PointData["L3_term"]

total_energy = mean_field_term + entropy_term + L1_term + L2_term + L3_term

output.PointData.append(total_energy, "total_energy")
