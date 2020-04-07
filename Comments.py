## Youtube comment extractor
# import statements
from oauth2client.tools import argparser
from apiclient.discovery import build
import json
import sys
import csv
import os
import time

# Constants
YOUTUBE_READ_WRITE_SCOPE = "https://www.googleapis.com/auth/youtube"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
DEVELOPER_KEY = "add-developer-key"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

## Utility Methods
def get_authenticated_service(args):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    return youtube

def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value)
                for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input    

def writeToCSVFile(listOfData, fileName):
  fileName = os.path.join(CURRENT_DIR,fileName)
  with open(fileName, 'ab') as f:
    writer = csv.writer(f, delimiter = "|")
    writer.writerows(listOfData)

## Video id extraction API
def youtube_search(youtube, options):
  # Call the search.list method to retrieve results matching the specified
  # query term.
  search_response = youtube.search().list(
    q='"' + options.q +'"'+'"OFFICIAL TRAILER"',
    part="id,snippet",
    maxResults=options.max_results
  ).execute()

  ## Store details as tuple (video_id, video_name, publish_date)
  videos = []
  channels = []
  playlists = []

  # Add each result to the appropriate list, and then display the lists of
  # matching videos, channels, and playlists.
  for search_result in search_response.get("items", []):
    #print search_response
    if search_result["id"]["kind"] == "youtube#video":
      newTuple = (byteify(search_result["id"]["videoId"]), 
                  byteify(search_result["snippet"]["title"]),
                  byteify(search_result["snippet"]["publishedAt"]))
      videos.append(newTuple)
    
  #for video in videos:
  #  print video   

  return videos 
    
## Extract Statistics for given video. Function takes input Video_id
def get_video_statistics(youtube, video_id):
  # Call video.list method to get details like statistics for each video_id
  video_response = youtube.videos().list(
    id=video_id,
    part='statistics'
  ).execute()

  # Declare variables
  commentCount = 0
  viewCount = 0
  favoriteCount = 0
  dislikeCount = 0
  likeCount = 0

  # Extract fields
  try:
    items = byteify(video_response['items'][0]['statistics'])
    commentCount = byteify(items['commentCount'])
    viewCount = byteify(items['viewCount'])
    favoriteCount = byteify(items['favoriteCount'])
    dislikeCount = byteify(items['dislikeCount'])
    likeCount = byteify(items['likeCount'])
  except:
    print("*** Error while extracting comments on video_id : "+video_id+" ***")
  # Make tuple and return
  finally:
    stats = (video_id, commentCount, viewCount, favoriteCount, dislikeCount, likeCount)
  return stats


## Comments extraction API
def get_comment_threads(youtube, video_id):
  results = youtube.commentThreads().list(
    part="snippet",
    videoId=video_id,
    textFormat="plainText"
  ).execute()

  # Handle Unicode
  byteify(results)

  #print(results)

  all_comments = []

  for item in results["items"]:
    local_list = []
    videoId = item["snippet"]["videoId"]
    comment = item["snippet"]["topLevelComment"]
    likeCount = comment["snippet"]["likeCount"]
    publishedAt = comment["snippet"]["publishedAt"]
    author = comment["snippet"]["authorDisplayName"]
    text = comment["snippet"]["textDisplay"]
    local_list.append(byteify(videoId))
    local_list.append(byteify(likeCount))
    local_list.append(byteify(publishedAt))
    local_list.append(byteify(author))
    local_list.append(byteify(text))
    all_comments.append(local_list)
    #print("\n videoId : {0}").format(byteify(videoId))
    #print("\n User : {0}").format(byteify(author))
    #print("\n publishedAt : {0}").format(byteify(publishedAt))
    #print("\n likeCount : {0}").format(byteify(likeCount))
    #print("\n Text : {0}").format(byteify(text))
    #print item
    #print "Comment by %s: %s" % (author, text)

  return all_comments

if __name__ == "__main__":
  
  print("\n Comments.py process starts......"+time.ctime())
  ## Generate the arguments
  argparser.add_argument("--videoid",help="Required; ID for video for which the comment will be inserted.")
  argparser.add_argument("--text", help="Required; text that will be used as comment.")
  argparser.add_argument("--q", help="Enter Movie keywords", default = "Spider Man")
  argparser.add_argument("--max-results", help="max results", default = 25)
  args = argparser.parse_args()

  ## Get authentication
  print("\n Authenticating with youtube API......"+time.ctime())
  youtube = get_authenticated_service(args)

  ## Get All video_id for given search term
  ## all_videos = [(video_id, video_name, publish_date)]
  print("\n Extracting all videos for query "+"'"+args.q+"'"+"......"+time.ctime())
  all_videos = youtube_search(youtube, args)

  ## Accumulators
  video_stats = []
  all_videos_data = []
  comments = []

  for video in all_videos:
    all_videos_data.append(list(video))

  ## Write all video information to file
  print("\n Writting Video Info file......"+time.ctime())
  writeToCSVFile(all_videos_data, "Videos_info.csv")

  ## Get comments for video_id
  print("\n Statistics & comments extration......"+time.ctime())
  for video in all_videos:
    video_id = video[0]
    # stats = (video_id, commentCount, viewCount, favoriteCount, dislikeCount, likeCount)
    stats = get_video_statistics(youtube, video_id)
    # Convert to list and append to parent list
    video_stats.append(list(stats))
    # comments = [videoid,likecount, publishedAt, author, text]
    comments = get_comment_threads(youtube, video_id)
    # Write all comments for a video to file in append mode.
    writeToCSVFile(comments, "Videos_comments.csv")
    #print(comments)

  # Write al video statistics to file
  print("\n Writting Video Stats file......"+time.ctime())
  writeToCSVFile(video_stats, "Videos_stats.csv")
  print("\n Comments.py process Ends......"+time.ctime())