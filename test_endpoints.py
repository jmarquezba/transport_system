import unittest
import json
import numpy as np
from webapp.app import app

class TestAppEndpoints(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['DEBUG'] = False
        self.client = app.test_client()

    def test_pages(self):
        """Verify main pages load correctly (status 200)"""
        for route in ['/', '/modulo1', '/modulo2', '/modulo3']:
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200, f"Route {route} failed with status {response.status_code}")
            self.assertIn(b'html', response.data.lower())

    def test_predict_demand_valid(self):
        """Verify LSTM real inference endpoint returns valid historical & forecast data"""
        payload = {"destination": "Taj Mahal"}
        response = self.client.post(
            '/api/predict_demand',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        
        # Verify keys
        expected_keys = ["destination", "historical", "forecast", "dates_hist", "dates_fore", "rmse", "mae", "mape"]
        for key in expected_keys:
            self.assertIn(key, data, f"Key {key} not found in response")
            
        # Verify sizes
        self.assertEqual(data["destination"], "Taj Mahal")
        self.assertEqual(len(data["historical"]), 30)
        self.assertEqual(len(data["forecast"]), 30)
        self.assertEqual(len(data["dates_hist"]), 30)
        self.assertEqual(len(data["dates_fore"]), 30)
        
        # Verify numeric types
        self.assertTrue(isinstance(data["rmse"], float) or isinstance(data["rmse"], int))
        self.assertTrue(isinstance(data["mae"], float) or isinstance(data["mae"], int))
        self.assertTrue(isinstance(data["mape"], float) or isinstance(data["mape"], int))
        
        # Verify values are floats
        for val in data["historical"]:
            self.assertTrue(isinstance(val, float) or isinstance(val, int))
        for val in data["forecast"]:
            self.assertTrue(isinstance(val, float) or isinstance(val, int))

    def test_predict_demand_invalid(self):
        """Verify invalid destination returns 400"""
        payload = {"destination": "Bogotá"}  # Invalid destination for new Kaggle model
        response = self.client.post(
            '/api/predict_demand',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn("error", data)

    def test_get_recommendations_valid(self):
        """Verify NCF real collaborative filtering returns top-5 custom recommendations"""
        payload = {"user_id": 327}  # Valid UserID in Kaggle dataset
        response = self.client.post(
            '/api/get_recommendations',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        
        # Verify keys
        self.assertIn("user_id", data)
        self.assertIn("recommendations", data)
        self.assertEqual(data["user_id"], 327)
        self.assertEqual(len(data["recommendations"]), 5)
        
        # Verify structure of first recommendation
        rec = data["recommendations"][0]
        self.assertEqual(rec["rank"], 1)
        self.assertIn("destination", rec)
        self.assertIn("category", rec)
        self.assertIn("country", rec)
        self.assertIn("cost_usd", rec)
        self.assertIn("score", rec)
        self.assertIn("description", rec)
        
        # Check that score decreases with rank
        scores = [r["score"] for r in data["recommendations"]]
        self.assertTrue(all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), "NCF recommendations are not sorted by score descending")

    def test_get_recommendations_invalid_user(self):
        """Verify invalid user ID returns 400 with a helpful list of valid IDs"""
        payload = {"user_id": 99999}  # Out of range
        response = self.client.post(
            '/api/get_recommendations',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data.decode('utf-8'))
        self.assertIn("error", data)
        self.assertIn("no existe en el dataset de Kaggle", data["error"])

if __name__ == '__main__':
    unittest.main()
