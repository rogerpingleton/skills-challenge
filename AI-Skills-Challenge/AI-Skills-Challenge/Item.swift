//
//  Item.swift
//  AI-Skills-Challenge
//
//  Created by Roger Pingleton on 4/15/26.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
