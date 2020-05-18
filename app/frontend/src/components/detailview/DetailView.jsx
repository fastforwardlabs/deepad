import React, { Component } from "react";
// import * as _ from "lodash"
import { Close16 } from "@carbon/icons-react"
import { Button } from 'carbon-components-react';
import "./detailview.css"


class DetailView extends Component {
    // constructor(props) {
    //     super(props)

    // }
    componentDidMount() {

    }

    closeButtonClick() {
        this.props.hideDetail()
    }



    render() {

        // console.log(this.props.explanation);

        let featureRows = this.props.dataDetails.map((data, index) => {
            if (data.feature === "id") {
                return (<div key={"hiddenrow" + index}></div>)
            } else {
                let cellId = data.id + ":" + data.feature
                return (
                    <div key={"explanation" + index} className="flex detailrow" style={{ backgroundColor: this.props.cellColors[cellId] }} >

                        <div className=" flex8  p10"> <span className="boldtext">{data.feature} </span>: {data.value}</div>
                        {/* <div className="p10 expdiv">  {data.explanation}</div> */}
                        <div className="flex2 "> </div>
                    </div>
                )
            }
        });

        return (
            <div>
                <div className="flex lightgreyhighlight ">
                    <div className="flexfull m10 boldtext "> {this.props.targetFeature} : {this.props.targetFeatureValue} </div>
                    <div className="">
                        <Button
                            onClick={this.closeButtonClick.bind(this)}
                            size={"field"}
                            renderIcon={Close16}
                            iconDescription={"."}
                        >
                            Back
                        </Button>
                    </div>

                </div>
                <div className="flex detailrow" >

                    <div className=" flex4  p10"> Feature </div>
                    {/* <div className="p10 expdiv">  Explanation Weight </div> */}
                    <div className="flex6 "> </div>
                </div>

                <div className="mt5">
                    {featureRows}
                </div>
            </div>


        );
    }
}

export default DetailView;