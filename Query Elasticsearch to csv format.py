import requests
import json
from elasticsearch import Elasticsearch
import csv

body_request={
  "aggs": {
    "2": {
      "date_histogram": {
        "field": "@timestamp",
        "interval": "1m",
        "time_zone": "Asia/Jakarta",
        "min_doc_count": 1
      }
    }
  },
  "size": 0,
  "_source": {
    "excludes": []
  },
  "stored_fields": [
    "*"
  ],
  "script_fields": {},
  "docvalue_fields": [
    {
      "field": "@timestamp",
      "format": "date_time"
    },
    {
      "field": "date",
      "format": "date_time"
    },
    {
      "field": "logstash_processed_at",
      "format": "date_time"
    },
    {
      "field": "new_timestamp",
      "format": "date_time"
    },
    {
      "field": "realtimestamp",
      "format": "date_time"
    }
  ],
  "query": {
    "bool": {
      "must": [
        {
          "match_all": {}
        },
        {
          "range": {
            "@timestamp": {
              "gte": "now-5m/m",
              "lte": "now/m",
              "format": "epoch_millis"
            }
          }
        }
      ],
      "filter": [],
      "should": [],
      "must_not": []
    }
  }
}


es = Elasticsearch('localhost:9200')
res = es.search(index="fortigate-*", body=body_request)
#print("%d events found" % res['hits']['total'])
#print(res)
i=0
jsonlen = (len(res['aggregations']['2']['buckets']))
#print(jsonlen)
with open('5mcount.csv', mode='w') as csv_file:
	fieldnames = ['timestamp', 'count']
	writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
	writer.writeheader()
	while i < jsonlen:

		timestamp_para = str(res['aggregations']['2']['buckets'][i]['key_as_string'])
		count_para = str(res['aggregations']['2']['buckets'][i]['doc_count'])
		writer.writerow({'timestamp': timestamp_para, 'count': count_para})
		#print(str(res['aggregations']['2']['buckets'][i]['key_as_string']) + "," + str(res['aggregations']['2']['buckets'][i]['doc_count']))
		i += 1

print("Export CSV Successfully")
