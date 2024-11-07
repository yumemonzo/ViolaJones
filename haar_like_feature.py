class RectangleRegion:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def compute_sum(self, integral_image):
        top_left_x = int(self.x)
        top_left_y = int(self.y)
        bottom_right_x = top_left_x + int(self.width) - 1
        bottom_right_y = top_left_y + int(self.height) - 1

        region_sum = int(integral_image[bottom_right_x, bottom_right_y])

        if top_left_x > 0:
            region_sum -= int(integral_image[top_left_x - 1, bottom_right_y])

        if top_left_y > 0:
            region_sum -= int(integral_image[bottom_right_x, top_left_y - 1])

        if top_left_x > 0 and top_left_y > 0:
            region_sum += int(integral_image[top_left_x - 1, top_left_y - 1])

        return region_sum


class HaarLikeFeature:
    def __init__(self, pos_region, neg_region):
        self.pos_region = pos_region
        self.neg_region = neg_region
    
    def compute_score(self, integral_image):
        sum_pos_region = sum([region.compute_sum(integral_image) for region in self.pos_region])
        sum_neg_region = sum([region.compute_sum(integral_image) for region in self.neg_region])
        
        return sum_neg_region - sum_pos_region
    

def build_haar_like_filters(image_width, image_height, shift=1, min_width=4, min_height=4):
    haar_like_filters = []

    def add_filter_if_within_bounds(main_regions, adjacent_regions, max_x, max_y):
        if max_x < image_width and max_y < image_height:
            haar_like_filters.append(HaarLikeFeature(main_regions, adjacent_regions))

    for filter_width in range(min_width, image_width + 1):
        for filter_height in range(min_height, image_height + 1):
            x = 0
            while x + filter_width < image_width:
                y = 0
                while y + filter_height < image_height:
                    main_region = RectangleRegion(x, y, filter_width, filter_height)
                    right_region = RectangleRegion(x + filter_width, y, filter_width, filter_height)
                    far_right_region = RectangleRegion(x + filter_width * 2, y, filter_width, filter_height)
                    bottom_region = RectangleRegion(x, y + filter_height, filter_width, filter_height)
                    bottom_right_region = RectangleRegion(x + filter_width, y + filter_height, filter_width, filter_height)

                    add_filter_if_within_bounds([main_region], [right_region], x + filter_width * 2, y)
                    add_filter_if_within_bounds([bottom_region], [main_region], x, y + filter_height * 2)
                    add_filter_if_within_bounds([main_region, far_right_region], [right_region], x + filter_width * 3, y)
                    add_filter_if_within_bounds([main_region, bottom_right_region], [bottom_region, right_region], x + filter_width * 2, y + filter_height * 2)

                    y += shift
                x += shift

    return haar_like_filters
